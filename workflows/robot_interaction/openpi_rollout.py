#!/usr/bin/env python3
"""Run an OpenPI policy against the existing Marvain HTTP robot driver."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config  # noqa: E402

from lerobot.robots.marvain_m6_http.config_marvain_m6_http import (  # noqa: E402
    HttpCameraConfig,
    MarvainM6HttpRobotConfig,
)
from lerobot.robots.marvain_m6_http.marvain_m6_http import MarvainM6HttpRobot  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


LOG = logging.getLogger("openpi_rollout")


def _build_robot(config: dict) -> MarvainM6HttpRobot:
    robot_cfg = config["robot"]
    if robot_cfg.get("type", "marvain_m6_http") != "marvain_m6_http":
        raise ValueError("OpenPI deployment currently requires robot.type=marvain_m6_http")

    cameras = {}
    for name, values in robot_cfg.get("cameras", {}).items():
        values = dict(values)
        camera_type = values.pop("type", "http")
        if camera_type != "http":
            raise ValueError(f"Camera {name!r} must use type=http, got {camera_type!r}")
        cameras[name] = HttpCameraConfig(**values)

    safety_path = robot_cfg.get("safety_stats_path")
    cfg = MarvainM6HttpRobotConfig(
        id=robot_cfg["id"],
        http_base_url=robot_cfg["http_base_url"],
        timeout=float(robot_cfg.get("timeout", 5.0)),
        action_chunk_path=robot_cfg.get("action_chunk_path", "/action_chunk"),
        cameras=cameras,
        joint_names=list(robot_cfg["joint_names"]),
        default_gripper_pos=float(robot_cfg.get("default_gripper_pos", 0.0)),
        safety_stats_path=Path(safety_path).expanduser() if safety_path else None,
        action_clip_margin_deg=float(robot_cfg.get("action_clip_margin_deg", 5.0)),
        max_relative_target_deg=float(robot_cfg.get("max_relative_target_deg", 10.0)),
        warn_on_observation_out_of_range=bool(
            robot_cfg.get("warn_on_observation_out_of_range", True)
        ),
    )
    return MarvainM6HttpRobot(cfg)


def _validate_config(config: dict) -> None:
    policy_cfg = config["openpi"]
    image_keys = policy_cfg["observation"]
    required_image_slots = {"image", "left_wrist_image", "right_wrist_image"}
    missing_slots = required_image_slots - set(image_keys)
    if missing_slots:
        raise ValueError(f"openpi.observation is missing image slots: {sorted(missing_slots)}")

    camera_names = set(config["robot"].get("cameras", {}))
    missing_cameras = set(image_keys.values()) - camera_names
    if missing_cameras:
        raise ValueError(f"OpenPI camera mapping references unknown cameras: {sorted(missing_cameras)}")

    joint_names = config["robot"].get("joint_names", [])
    if len(joint_names) != 16 or len(set(joint_names)) != 16:
        raise ValueError("OpenPI requires exactly 16 unique joint_names")

    inference = config["inference"]
    n_action_steps = int(inference.get("n_action_steps", 30))
    if not 1 <= n_action_steps <= 50:
        raise ValueError("inference.n_action_steps must be in [1, 50] for pi0.5")
    if float(inference.get("fps", 30.0)) <= 0:
        raise ValueError("inference.fps must be positive")


def _observation_for_policy(config: dict, robot_obs: dict) -> dict:
    image_keys = config["openpi"]["observation"]
    missing = [name for name in image_keys.values() if name not in robot_obs]
    if missing:
        raise RuntimeError(f"Robot observation is missing OpenPI cameras: {missing}")

    joint_names = config["robot"]["joint_names"]
    state = np.asarray([robot_obs[f"{name}.pos"] for name in joint_names], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid 16-D robot state: shape={state.shape}, finite={np.isfinite(state).all()}")

    return {
        "image": np.asarray(robot_obs[image_keys["image"]], dtype=np.uint8),
        "left_wrist_image": np.asarray(
            robot_obs[image_keys["left_wrist_image"]], dtype=np.uint8
        ),
        "right_wrist_image": np.asarray(
            robot_obs[image_keys["right_wrist_image"]], dtype=np.uint8
        ),
        "state": state,
        "prompt": config["dataset"]["single_task"],
    }


def _actions_for_robot(config: dict, result: dict) -> list[dict[str, float]]:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise RuntimeError(f"OpenPI returned invalid action shape {actions.shape}; expected [T, 16]")
    if not np.isfinite(actions).all():
        raise RuntimeError("OpenPI returned NaN or Inf actions")

    n_action_steps = min(int(config["inference"].get("n_action_steps", 30)), len(actions))
    selected = actions[:n_action_steps].copy()
    openpi_cfg = config["openpi"]
    selected[:, 14:16] = np.clip(
        selected[:, 14:16],
        float(openpi_cfg.get("gripper_min", 0.0)),
        float(openpi_cfg.get("gripper_max", 1.0)),
    )

    joint_names = config["robot"]["joint_names"]
    return [
        {f"{name}.pos": float(value) for name, value in zip(joint_names, row, strict=True)}
        for row in selected
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--http-base-url", default=None)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--safety-stats-path", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.http_base_url is not None:
        config["robot"]["http_base_url"] = args.http_base_url
    if args.robot_id is not None:
        config["robot"]["id"] = args.robot_id
    if args.safety_stats_path is not None:
        config["robot"]["safety_stats_path"] = args.safety_stats_path
    if args.task is not None:
        config["dataset"]["single_task"] = args.task
    if args.fps is not None:
        config["inference"]["fps"] = args.fps
    if args.duration is not None:
        config["inference"]["duration"] = args.duration
    if args.n_action_steps is not None:
        config["inference"]["n_action_steps"] = args.n_action_steps
    _validate_config(config)
    robot = _build_robot(config)
    if args.validate_only:
        LOG.info("OpenPI rollout configuration is valid; no robot connection was made")
        return 0

    policy_cfg = config["openpi"]
    host = args.policy_host or policy_cfg.get("server_host", "127.0.0.1")
    port = args.policy_port or int(policy_cfg.get("server_port", 8000))
    LOG.info("Connecting to OpenPI policy server at ws://%s:%d", host, port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    LOG.info("OpenPI server metadata: %s", policy.get_server_metadata())

    stop_requested = False

    def request_stop(signum: int, _frame) -> None:
        nonlocal stop_requested
        LOG.info("Received signal %d; stopping after the current safe operation", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)

    fps = float(config["inference"].get("fps", 30.0))
    duration = float(config["inference"].get("duration", 0.0))
    started_at = time.monotonic()
    iteration = 0

    LOG.info("Connecting Marvain HTTP robot only after policy readiness checks passed")
    robot.connect()
    try:
        while not stop_requested:
            if duration > 0 and time.monotonic() - started_at >= duration:
                break

            cycle_started = time.monotonic()
            observation = _observation_for_policy(config, robot.get_observation())
            inference_started = time.monotonic()
            result = policy.infer(observation)
            inference_s = time.monotonic() - inference_started
            actions = _actions_for_robot(config, result)
            robot.send_action_chunk(actions)

            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                flat = np.asarray(
                    [[action[f"{name}.pos"] for name in config["robot"]["joint_names"]] for action in actions]
                )
                LOG.info(
                    "chunk=%d steps=%d inference=%.3fs action_range=[%.4f, %.4f]",
                    iteration,
                    len(actions),
                    inference_s,
                    float(flat.min()),
                    float(flat.max()),
                )

            target_cycle_s = len(actions) / fps
            remaining = target_cycle_s - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        robot.disconnect()
        LOG.info("Robot disconnected cleanly")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a loopback InternVLA policy service against the Marvain HTTP robot."""

from __future__ import annotations

import argparse
import http.client
import io
import json
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


LOG = logging.getLogger("internvla_rollout")


def build_robot(config: dict) -> MarvainM6HttpRobot:
    robot_cfg = config["robot"]
    if robot_cfg.get("type", "marvain_m6_http") != "marvain_m6_http":
        raise ValueError("InternVLA deployment requires robot.type=marvain_m6_http")

    cameras = {}
    for name, values in robot_cfg.get("cameras", {}).items():
        values = dict(values)
        camera_type = values.pop("type", "http")
        if camera_type != "http":
            raise ValueError(f"Camera {name!r} must use type=http, got {camera_type!r}")
        cameras[name] = HttpCameraConfig(**values)

    safety_path = robot_cfg.get("safety_stats_path")
    robot_config = MarvainM6HttpRobotConfig(
        id=robot_cfg["id"],
        http_base_url=robot_cfg["http_base_url"],
        timeout=float(robot_cfg.get("timeout", 5.0)),
        action_chunk_path=robot_cfg.get("action_chunk_path", "/action_chunk"),
        cameras=cameras,
        joint_names=list(robot_cfg["joint_names"]),
        default_gripper_pos=float(robot_cfg.get("default_gripper_pos", 0.0)),
        safety_stats_path=Path(safety_path).expanduser() if safety_path else None,
        action_clip_margin_deg=float(robot_cfg.get("action_clip_margin_deg", 0.05)),
        max_relative_target_deg=float(robot_cfg.get("max_relative_target_deg", 0.08)),
        warn_on_observation_out_of_range=bool(
            robot_cfg.get("warn_on_observation_out_of_range", True)
        ),
    )
    return MarvainM6HttpRobot(robot_config)


def validate_config(config: dict) -> None:
    names = config["robot"].get("joint_names", [])
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("InternVLA deployment requires exactly 16 unique joint_names")
    mapping = config["internvla"].get("observation", {})
    required = {"image0", "image1", "image2"}
    if set(mapping) != required:
        raise ValueError(f"internvla.observation must define exactly {sorted(required)}")
    missing_cameras = set(mapping.values()) - set(config["robot"].get("cameras", {}))
    if missing_cameras:
        raise ValueError(f"InternVLA camera mapping references unknown cameras: {missing_cameras}")
    n_action_steps = int(config["inference"].get("n_action_steps", 30))
    if not 1 <= n_action_steps <= 50:
        raise ValueError("inference.n_action_steps must be in [1, 50]")
    task = config["dataset"].get("single_task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("dataset.single_task must be a non-empty string")


class InternVLAPolicyClient:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.connection: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self.connection is None:
            self.connection = http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=self.timeout_s,
            )
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[str, bytes]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"InternVLA service returned HTTP {response.status}: "
                        f"{response_body.decode('utf-8', errors='replace')}"
                    )
                return response.getheader("Content-Type", ""), response_body
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
                self.close()
                if attempt == 0:
                    continue
        raise RuntimeError("Unable to communicate with InternVLA policy service") from last_error

    def health(self) -> dict:
        content_type, body = self._request("GET", "/healthz")
        if "application/json" not in content_type:
            raise RuntimeError(f"Unexpected InternVLA health content type: {content_type}")
        payload = json.loads(body.decode("utf-8"))
        if payload.get("status") != "ready" or payload.get("model") != "internvla_a1_5":
            raise RuntimeError(f"Unexpected InternVLA health response: {payload}")
        return payload

    def infer(self, config: dict, robot_obs: dict) -> np.ndarray:
        mapping = config["internvla"]["observation"]
        missing = [name for name in mapping.values() if name not in robot_obs]
        if missing:
            raise RuntimeError(f"Robot observation is missing InternVLA cameras: {missing}")
        names = config["robot"]["joint_names"]
        state = np.asarray([robot_obs[f"{name}.pos"] for name in names], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise RuntimeError(f"Invalid robot state: shape={state.shape}")

        request = io.BytesIO()
        np.savez(
            request,
            state=state,
            image0=np.asarray(robot_obs[mapping["image0"]], dtype=np.uint8),
            image1=np.asarray(robot_obs[mapping["image1"]], dtype=np.uint8),
            image2=np.asarray(robot_obs[mapping["image2"]], dtype=np.uint8),
            prompt=np.asarray(config["dataset"]["single_task"]),
        )
        content_type, body = self._request(
            "POST",
            "/infer",
            body=request.getvalue(),
            content_type="application/x-npz",
        )
        if "application/x-npz" not in content_type:
            raise RuntimeError(f"Unexpected InternVLA inference content type: {content_type}")
        with np.load(io.BytesIO(body), allow_pickle=False) as response:
            if set(response.files) != {"actions"}:
                raise RuntimeError(f"Unexpected InternVLA response fields: {response.files}")
            actions = np.asarray(response["actions"], dtype=np.float32)
        expected_steps = int(config["inference"].get("n_action_steps", 30))
        if actions.shape != (expected_steps, 16) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid InternVLA actions: shape={actions.shape}")
        return actions


def actions_for_robot(config: dict, actions: np.ndarray) -> list[dict[str, float]]:
    names = config["robot"]["joint_names"]
    return [
        {f"{name}.pos": float(value) for name, value in zip(names, row, strict=True)}
        for row in actions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    validate_config(config)
    robot = build_robot(config)
    if args.validate_only:
        LOG.info("InternVLA robot client configuration is valid; no connection was made")
        return 0

    runtime = config["internvla"]
    host = args.policy_host or runtime.get("server_host", "127.0.0.1")
    port = args.policy_port or int(runtime.get("server_port", 8001))
    client = InternVLAPolicyClient(
        host,
        port,
        float(runtime.get("request_timeout_s", 300.0)),
    )
    metadata = client.health()
    LOG.info("InternVLA service metadata: %s", metadata)

    stop_requested = False

    def request_stop(signum: int, _frame) -> None:
        nonlocal stop_requested
        LOG.info("Received signal %d; stopping safely", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    fps = float(config["inference"].get("fps", 30.0))
    duration = float(config["inference"].get("duration", 0.0))
    max_steps = int(config["inference"].get("max_steps", 10000))
    started_at = time.monotonic()
    iteration = 0

    LOG.info("Connecting to the robot after InternVLA service readiness")
    robot.connect()
    try:
        while not stop_requested and iteration < max_steps:
            if duration > 0 and time.monotonic() - started_at >= duration:
                break
            cycle_started = time.monotonic()
            inference_started = time.monotonic()
            actions = client.infer(config, robot.get_observation())
            inference_s = time.monotonic() - inference_started
            robot_actions = actions_for_robot(config, actions)
            robot.send_action_chunk(robot_actions)

            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                LOG.info(
                    "chunk=%d steps=%d inference=%.3fs action_range=[%.4f, %.4f]",
                    iteration,
                    len(actions),
                    inference_s,
                    float(actions.min()),
                    float(actions.max()),
                )
            remaining = len(actions) / fps - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            robot.disconnect()
            LOG.info("Robot disconnected cleanly")
        finally:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

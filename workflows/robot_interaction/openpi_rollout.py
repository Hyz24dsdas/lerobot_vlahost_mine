#!/usr/bin/env python3
"""Run an OpenPI policy against the existing Marvain HTTP robot driver."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config  # noqa: E402

from lerobot.robots.marvain_m6_http.config_marvain_m6_http import (  # noqa: E402
    HttpCameraConfig,
    MarvainM6HttpRobotConfig,
)
from lerobot.robots.marvain_m6_http.marvain_m6_http import MarvainM6HttpRobot  # noqa: E402
from openpi_rtc import RTCRuntimeConfig, RTCRequestContext, RTCScheduler  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


LOG = logging.getLogger("openpi_rollout")


class InterventionSession:
    """Keep VLAHost's foot-intervention ownership alive for this rollout."""

    def __init__(self, config: dict):
        cfg = config.get("intervention", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.required = bool(cfg.get("required", True))
        base_url = str(config["robot"]["http_base_url"]).rstrip("/")
        self.url = base_url + str(cfg.get("session_path", "/intervention/session"))
        self.heartbeat_sec = max(0.2, float(cfg.get("heartbeat_sec", 0.5)))
        self.timeout_sec = max(0.2, float(cfg.get("timeout_sec", 2.0)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _post(self, enabled: bool) -> bool:
        try:
            response = requests.post(
                self.url,
                json={"enabled": enabled},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            result = response.json()
            return bool(result.get("success", False))
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("Intervention session update failed: %s", exc)
            return False

    def start(self) -> None:
        if not self.enabled:
            return
        if not self._post(True):
            if self.required:
                raise RuntimeError(
                    "VLAHost intervention session could not start; refusing robot control"
                )
            return
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="openpi-intervention-heartbeat",
            daemon=True,
        )
        self._thread.start()
        LOG.info("Foot intervention coordination enabled at %s", self.url)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_sec):
            self._post(True)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_sec + 0.5)
            self._thread = None
        self._post(False)


class PromptStageController:
    """Manage a one-way prompt transition without touching the policy server."""

    def __init__(self, config: dict):
        staged_cfg = config.get("staged_task", {})
        self.enabled = bool(staged_cfg.get("enabled", False))
        self._switch_requested = threading.Event()
        self._current_stage = "default"
        self._switch_input = "1"
        self._prompts = {"default": str(config["dataset"]["single_task"])}

        if not self.enabled:
            return

        initial_stage = str(staged_cfg.get("initial_stage", "spread"))
        if initial_stage != "spread":
            raise ValueError("staged_task.initial_stage must be 'spread'")

        stages = staged_cfg.get("stages", {})
        try:
            spread_prompt = str(stages["spread"]["prompt"]).strip()
            fold_prompt = str(stages["fold"]["prompt"]).strip()
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "staged_task.stages must define spread.prompt and fold.prompt"
            ) from exc
        if not spread_prompt or not fold_prompt:
            raise ValueError("staged_task prompts must be non-empty")
        if spread_prompt == fold_prompt:
            raise ValueError("spread and fold prompts must be different")
        if str(config["dataset"]["single_task"]) != spread_prompt:
            raise ValueError(
                "dataset.single_task must exactly match staged_task.stages.spread.prompt"
            )

        self._switch_input = str(staged_cfg.get("switch_input", "1")).strip()
        if not self._switch_input:
            raise ValueError("staged_task.switch_input must be non-empty")
        self._prompts = {"spread": spread_prompt, "fold": fold_prompt}
        self._current_stage = initial_stage

    @property
    def current_stage(self) -> str:
        return self._current_stage

    @property
    def prompt(self) -> str:
        return self._prompts[self._current_stage]

    def start_input_listener(self) -> None:
        if not self.enabled:
            return
        listener = threading.Thread(
            target=self._read_input,
            name="openpi-prompt-stage-input",
            daemon=True,
        )
        listener.start()
        LOG.info(
            "Stage=spread. Enter %s and press Enter once the garment is fully spread; "
            "the next inference will use the fold prompt.",
            self._switch_input,
        )

    def _read_input(self) -> None:
        while self._current_stage == "spread":
            line = sys.stdin.readline()
            if line == "":
                LOG.warning("Stage-switch input is unavailable because stdin was closed")
                return
            value = line.strip()
            if value == self._switch_input:
                self._switch_requested.set()
                LOG.info("Fold-stage switch requested; waiting for the current control cycle")
                return
            if value:
                LOG.warning(
                    "Ignoring stage input %r; enter %s to switch to fold",
                    value,
                    self._switch_input,
                )

    def apply_pending_switch(self) -> bool:
        if (
            not self.enabled
            or self._current_stage != "spread"
            or not self._switch_requested.is_set()
        ):
            return False
        self._current_stage = "fold"
        self._switch_requested.clear()
        return True


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
    PromptStageController(config)
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
    inference_type = str(inference.get("type", "chunk")).lower()
    if inference_type not in {"chunk", "rtc"}:
        raise ValueError("inference.type must be 'chunk' or 'rtc' for native OpenPI")
    n_action_steps = int(inference.get("n_action_steps", 30))
    if not 1 <= n_action_steps <= 50:
        raise ValueError("inference.n_action_steps must be in [1, 50] for pi0.5")
    if float(inference.get("fps", 30.0)) <= 0:
        raise ValueError("inference.fps must be positive")
    if inference_type == "rtc":
        RTCRuntimeConfig.from_mapping(inference.get("rtc"))
    intervention = config.get("intervention", {})
    if intervention.get("enabled", False):
        heartbeat_sec = float(intervention.get("heartbeat_sec", 0.5))
        if heartbeat_sec <= 0.0:
            raise ValueError("intervention.heartbeat_sec must be positive")


def _observation_for_policy(config: dict, robot_obs: dict, *, prompt: str | None = None) -> dict:
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
        "prompt": prompt if prompt is not None else config["dataset"]["single_task"],
    }


def _actions_for_robot(
    config: dict,
    result: dict,
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> list[dict[str, float]]:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise RuntimeError(f"OpenPI returned invalid action shape {actions.shape}; expected [T, 16]")
    if not np.isfinite(actions).all():
        raise RuntimeError("OpenPI returned NaN or Inf actions")

    if end_step is None:
        end_step = start_step + int(config["inference"].get("n_action_steps", 30))
    start_step = max(0, min(int(start_step), len(actions)))
    end_step = max(start_step, min(int(end_step), len(actions)))
    selected = actions[start_step:end_step].copy()
    if not len(selected):
        raise RuntimeError(
            f"OpenPI action selection is empty: start={start_step}, end={end_step}, "
            f"returned={len(actions)}"
        )
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
    parser.add_argument("--inference-type", choices=["chunk", "rtc"], default=None)
    parser.add_argument("--rtc-execution-horizon", type=int, default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["LINEAR", "EXP", "ONES", "ZEROS"],
        default=None,
    )
    parser.add_argument("--rtc-queue-threshold", type=int, default=None)
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
    if args.inference_type is not None:
        config["inference"]["type"] = args.inference_type
    rtc_overrides = config["inference"].setdefault("rtc", {})
    if args.rtc_execution_horizon is not None:
        rtc_overrides["execution_horizon"] = args.rtc_execution_horizon
    if args.rtc_max_guidance_weight is not None:
        rtc_overrides["max_guidance_weight"] = args.rtc_max_guidance_weight
    if args.rtc_prefix_attention_schedule is not None:
        rtc_overrides["prefix_attention_schedule"] = args.rtc_prefix_attention_schedule
    if args.rtc_queue_threshold is not None:
        rtc_overrides["queue_threshold"] = args.rtc_queue_threshold
    _validate_config(config)
    stage_controller = PromptStageController(config)
    intervention_session = InterventionSession(config)
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
    inference_type = str(config["inference"].get("type", "chunk")).lower()
    rtc_scheduler = None
    if inference_type == "rtc":
        rtc_scheduler = RTCScheduler(
            RTCRuntimeConfig.from_mapping(config["inference"].get("rtc")),
            fps,
        )
    started_at = time.monotonic()
    iteration = 0

    LOG.info(
        "Initial prompt stage=%s prompt=%r",
        stage_controller.current_stage,
        stage_controller.prompt,
    )
    if rtc_scheduler is None:
        LOG.info(
            "OpenPI inference mode=chunk: dispatching %d steps every %.3fs",
            int(config["inference"].get("n_action_steps", 30)),
            float(config["inference"].get("chunk_interval_s", 1.0)),
        )
    else:
        LOG.info("OpenPI inference mode=rtc: %s", rtc_scheduler.config)
    LOG.info("Connecting Marvain HTTP robot only after policy readiness checks passed")
    intervention_session.start()
    try:
        robot.connect()
        try:
            while not stop_requested:
                if duration > 0 and time.monotonic() - started_at >= duration:
                    break

                if rtc_scheduler is not None:
                    while not stop_requested:
                        now = time.monotonic()
                        if duration > 0 and now - started_at >= duration:
                            stop_requested = True
                            break
                        wait_s = rtc_scheduler.seconds_until_inference(now)
                        if wait_s <= 0.0:
                            break
                        time.sleep(min(wait_s, 0.05))
                    if stop_requested:
                        break

                cycle_started = time.monotonic()
                robot_observation = robot.get_observation()
                stage_switched = stage_controller.apply_pending_switch()
                if stage_switched:
                    robot.sync_action_reference(robot_observation)
                    if rtc_scheduler is not None:
                        rtc_scheduler.reset_prefix()
                    LOG.info(
                        "Stage switched to fold. Pending spread actions will be replaced by the next chunk; "
                        "prompt=%r",
                        stage_controller.prompt,
                    )
                observation = _observation_for_policy(
                    config,
                    robot_observation,
                    prompt=stage_controller.prompt,
                )
                inference_started = time.monotonic()
                rtc_request: RTCRequestContext | None = None
                if rtc_scheduler is not None:
                    rtc_request = rtc_scheduler.build_request(
                        inference_started,
                        reset=stage_switched,
                    )
                    observation["_rtc"] = rtc_request.payload
                result = policy.infer(observation)
                inference_s = time.monotonic() - inference_started
                rtc_window = None
                if rtc_scheduler is None:
                    actions = _actions_for_robot(config, result)
                else:
                    returned_actions = np.asarray(result.get("actions"))
                    rtc_window = rtc_scheduler.action_window(
                        len(returned_actions),
                        inference_s,
                        rtc_request,
                    )
                    # RTC preempts the unexecuted tail of the previous chunk.
                    # Re-anchor per-step safety limiting to measured state instead
                    # of that previously submitted, but never executed, tail.
                    robot.sync_action_reference(robot_observation)
                    actions = _actions_for_robot(
                        config,
                        result,
                        start_step=rtc_window.start_step,
                        end_step=rtc_window.end_step,
                    )
                robot.send_action_chunk(actions, source_hz=fps)
                sent_at = time.monotonic()
                receipt = robot.last_chunk_metadata or {}
                if rtc_scheduler is not None:
                    rtc_response = result.get("rtc")
                    if not isinstance(rtc_response, dict):
                        raise RuntimeError(
                            "Native OpenPI server did not return RTC metadata; "
                            "verify the RTC-enabled OpenPI policy patch is installed"
                        )
                    rtc_scheduler.record_dispatch(
                        rtc_response=rtc_response,
                        request=rtc_request,
                        round_trip_s=inference_s,
                        window=rtc_window,
                        execution=str(receipt.get("execution", "unknown")),
                        sent_at=sent_at,
                    )

                iteration += 1
                if iteration == 1:
                    stage_controller.start_input_listener()
                should_log = iteration == 1 or iteration % 10 == 0
                if rtc_scheduler is not None:
                    should_log = iteration <= 5 or iteration % 10 == 0
                if should_log:
                    flat = np.asarray(
                        [
                            [
                                action[f"{name}.pos"]
                                for name in config["robot"]["joint_names"]
                            ]
                            for action in actions
                        ]
                    )
                    model_s = float(result.get("policy_timing", {}).get("infer_ms", 0.0)) / 1000.0
                    if rtc_scheduler is None:
                        LOG.info(
                            "chunk=%d server_chunk=%s execution=%s stage=%s steps=%d "
                            "inference=%.3fs model=%.3fs action_range=[%.4f, %.4f]",
                            iteration,
                            receipt.get("chunk_id", "?"),
                            receipt.get("execution", "?"),
                            stage_controller.current_stage,
                            len(actions),
                            inference_s,
                            model_s,
                            float(flat.min()),
                            float(flat.max()),
                        )
                    else:
                        rtc_response = result["rtc"]
                        LOG.info(
                            "rtc_chunk=%d policy_chunk=%s server_chunk=%s execution=%s "
                            "stage=%s sent_steps=%d stale_steps=%d estimated_delay=%d "
                            "prefix=%s prefix_start=%d inference=%.3fs model=%.3fs "
                            "action_range=[%.4f, %.4f]",
                            iteration,
                            rtc_response.get("chunk_id", "?"),
                            receipt.get("chunk_id", "?"),
                            receipt.get("execution", "?"),
                            stage_controller.current_stage,
                            len(actions),
                            rtc_window.start_step,
                            rtc_request.estimated_delay_steps,
                            rtc_response.get("prefix_applied", False),
                            rtc_request.prefix_start_step,
                            inference_s,
                            model_s,
                            float(flat.min()),
                            float(flat.max()),
                        )

                if rtc_scheduler is None:
                    target_cycle_s = len(actions) / fps
                    remaining = target_cycle_s - (time.monotonic() - cycle_started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if robot.is_connected:
                robot.disconnect()
                LOG.info("Robot disconnected cleanly")
    finally:
        intervention_session.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

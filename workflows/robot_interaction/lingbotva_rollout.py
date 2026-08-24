#!/usr/bin/env python3
"""Run an external LingBot-VA policy against the Marvain HTTP robot."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config  # noqa: E402

from lerobot.robots.marvain_m6_http.config_marvain_m6_http import (  # noqa: E402
    HttpCameraConfig,
    MarvainM6HttpRobotConfig,
)
from lerobot.robots.marvain_m6_http.marvain_m6_http import MarvainM6HttpRobot  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


LOG = logging.getLogger("lingbotva_rollout")


@dataclass(frozen=True)
class StutterOptions:
    enabled: bool
    mode: str
    playback_fps: float
    adaptive_playback: bool
    min_playback_fps: float
    max_playback_fps: float
    latency_safety_factor: float
    latency_ema_alpha: float
    log_every_n_frames: int

    @classmethod
    def from_config(cls, config: dict) -> "StutterOptions":
        values = config["inference"].get("stutter_optimization", {})
        enabled = bool(values.get("enabled", False))
        mode = str(values.get("mode", "sync")) if enabled else "sync"
        result = cls(
            enabled=enabled,
            mode=mode,
            playback_fps=float(values.get("playback_fps", config["inference"].get("fps", 30.0))),
            adaptive_playback=bool(values.get("adaptive_playback", True)),
            min_playback_fps=float(values.get("min_playback_fps", 5.0)),
            max_playback_fps=float(values.get("max_playback_fps", 30.0)),
            latency_safety_factor=float(values.get("latency_safety_factor", 1.15)),
            latency_ema_alpha=float(values.get("latency_ema_alpha", 0.25)),
            log_every_n_frames=int(values.get("log_every_n_frames", 1)),
        )
        if result.mode not in {"sync", "predictive_overlap"}:
            raise ValueError(
                "inference.stutter_optimization.mode must be sync or predictive_overlap"
            )
        for name in ("playback_fps", "min_playback_fps", "max_playback_fps"):
            value = getattr(result, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"stutter_optimization.{name} must be finite and positive")
        if result.min_playback_fps > result.max_playback_fps:
            raise ValueError(
                "stutter_optimization.min_playback_fps must not exceed max_playback_fps"
            )
        if not result.min_playback_fps <= result.playback_fps <= result.max_playback_fps:
            raise ValueError(
                "stutter_optimization.playback_fps must be within the configured bounds"
            )
        if result.latency_safety_factor < 1.0:
            raise ValueError("stutter_optimization.latency_safety_factor must be at least 1.0")
        if not 0.0 < result.latency_ema_alpha <= 1.0:
            raise ValueError("stutter_optimization.latency_ema_alpha must be in (0, 1]")
        if result.log_every_n_frames <= 0:
            raise ValueError("stutter_optimization.log_every_n_frames must be positive")
        return result


@dataclass
class FrameExecution:
    model_actions: np.ndarray
    robot_actions: list[dict[str, float]]
    executed_actions: list[dict[str, float]]
    observations: list[dict]
    final_observation: dict
    elapsed_s: float


@dataclass
class ModelCycle:
    model_actions: np.ndarray
    robot_actions: list[dict[str, float]]
    cache_s: float
    inference_s: float
    elapsed_s: float


def build_robot(config: dict) -> MarvainM6HttpRobot:
    robot_cfg = config["robot"]
    if robot_cfg.get("type", "marvain_m6_http") != "marvain_m6_http":
        raise ValueError("LingBot-VA deployment requires robot.type=marvain_m6_http")

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
        raise ValueError("LingBot-VA deployment requires exactly 16 unique joint_names")

    runtime = config["lingbotva"]
    mapping = runtime.get("observation", {})
    required = {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    if set(mapping) != required:
        raise ValueError(f"lingbotva.observation must define exactly {sorted(required)}")
    missing_cameras = set(mapping.values()) - set(config["robot"].get("cameras", {}))
    if missing_cameras:
        raise ValueError(f"LingBot-VA camera mapping references unknown cameras: {missing_cameras}")

    action_per_frame = int(runtime.get("action_per_frame", 16))
    frame_chunk_size = int(runtime.get("frame_chunk_size", 2))
    temporal_compression = int(runtime.get("vae_temporal_compression", 4))
    n_action_steps = int(config["inference"].get("n_action_steps", 32))
    expected_horizon = action_per_frame * frame_chunk_size
    if action_per_frame <= 0 or frame_chunk_size <= 0 or temporal_compression <= 0:
        raise ValueError(
            "LingBot-VA action_per_frame, frame_chunk_size, and "
            "vae_temporal_compression must be positive"
        )
    if action_per_frame % temporal_compression:
        raise ValueError(
            "LingBot-VA action_per_frame must be divisible by "
            "vae_temporal_compression"
        )
    if not 1 <= n_action_steps <= expected_horizon:
        raise ValueError("inference.n_action_steps exceeds the LingBot-VA action horizon")
    if n_action_steps % action_per_frame:
        raise ValueError("inference.n_action_steps must be divisible by action_per_frame")
    stutter = StutterOptions.from_config(config)
    if stutter.mode == "predictive_overlap" and n_action_steps != expected_horizon:
        raise ValueError(
            "predictive_overlap requires inference.n_action_steps to equal the full "
            f"LingBot-VA horizon ({expected_horizon})"
        )
    if runtime.get("model_joint_unit") != "degrees":
        raise ValueError("This trained LingBot-VA checkpoint must declare model_joint_unit=degrees")
    if runtime.get("robot_joint_unit") != "radians":
        raise ValueError("The Marvain VLAHost endpoint must declare robot_joint_unit=radians")

    task = config["dataset"].get("single_task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("dataset.single_task must be a non-empty string")


def observation_for_policy(config: dict, robot_obs: dict) -> dict:
    mapping = config["lingbotva"]["observation"]
    missing = [camera for camera in mapping.values() if camera not in robot_obs]
    if missing:
        raise RuntimeError(f"Robot observation is missing LingBot-VA cameras: {missing}")
    return {
        model_key: np.asarray(robot_obs[camera], dtype=np.uint8)
        for model_key, camera in mapping.items()
    } | {"prompt": config["dataset"]["single_task"]}


def model_actions_to_robot(
    config: dict,
    result: dict,
    *,
    first_chunk: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise RuntimeError(
            f"LingBot-VA returned invalid action shape {actions.shape}; expected [T, 16]"
        )
    if not np.isfinite(actions).all():
        raise RuntimeError("LingBot-VA returned NaN or Inf actions")

    runtime = config["lingbotva"]
    action_per_frame = int(runtime.get("action_per_frame", 16))
    frame_chunk_size = int(runtime.get("frame_chunk_size", 2))
    expected_horizon = action_per_frame * frame_chunk_size
    if len(actions) != expected_horizon:
        raise RuntimeError(
            f"LingBot-VA returned {len(actions)} steps; expected {expected_horizon}"
        )

    # The first predicted video frame is conditioned on the initial observation.
    # Its actions belong in the KV cache but must not be sent to the robot.
    start_index = action_per_frame if first_chunk else 0
    requested_steps = int(config["inference"].get("n_action_steps", 32))
    execute_count = min(requested_steps, len(actions) - start_index)
    execute_count -= execute_count % action_per_frame
    if execute_count <= 0:
        raise RuntimeError(
            "LingBot-VA output does not contain a complete executable action frame"
        )
    selected = actions[start_index : start_index + execute_count].copy()
    cache_prefix = actions[:start_index].copy()
    arm_degrees = np.concatenate((selected[:, :7], selected[:, 8:15]), axis=1)
    max_abs_joint_deg = float(runtime.get("max_abs_joint_deg", 180.0))
    if np.abs(arm_degrees).max() > max_abs_joint_deg:
        raise RuntimeError(
            f"LingBot-VA arm output exceeds {max_abs_joint_deg:.1f} degrees; refusing to send"
        )

    robot_values = np.empty_like(selected)
    robot_values[:, :14] = np.deg2rad(arm_degrees)
    gripper_min = float(runtime.get("gripper_min", 0.0))
    gripper_max = float(runtime.get("gripper_max", 1.0))
    robot_values[:, 14] = np.clip(selected[:, 7], gripper_min, gripper_max)
    robot_values[:, 15] = np.clip(selected[:, 15], gripper_min, gripper_max)

    names = config["robot"]["joint_names"]
    robot_actions = [
        {f"{name}.pos": float(value) for name, value in zip(names, row, strict=True)}
        for row in robot_values
    ]
    return selected, cache_prefix, robot_actions


def actual_robot_actions_to_model(config: dict, actual_actions: list[dict[str, float]]) -> np.ndarray:
    names = config["robot"]["joint_names"]
    rows = []
    for action in actual_actions:
        robot_row = np.asarray([action[f"{name}.pos"] for name in names], dtype=np.float32)
        rows.append(
            np.concatenate(
                (
                    np.rad2deg(robot_row[:7]),
                    robot_row[14:15],
                    np.rad2deg(robot_row[7:14]),
                    robot_row[15:16],
                )
            )
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 16 or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid executed action data for LingBot-VA cache: {result.shape}")
    return result


def split_action_frames(
    config: dict,
    model_actions: np.ndarray,
    robot_actions: list[dict[str, float]],
) -> list[tuple[np.ndarray, list[dict[str, float]]]]:
    action_per_frame = int(config["lingbotva"].get("action_per_frame", 16))
    if len(model_actions) != len(robot_actions) or len(model_actions) % action_per_frame:
        raise RuntimeError(
            "LingBot-VA action plan cannot be split into complete aligned frames: "
            f"model={len(model_actions)}, robot={len(robot_actions)}, "
            f"action_per_frame={action_per_frame}"
        )
    return [
        (
            model_actions[offset : offset + action_per_frame].copy(),
            robot_actions[offset : offset + action_per_frame],
        )
        for offset in range(0, len(robot_actions), action_per_frame)
    ]


def execute_action_frame(
    config: dict,
    robot: MarvainM6HttpRobot,
    model_actions: np.ndarray,
    robot_actions: list[dict[str, float]],
    playback_fps: float,
) -> FrameExecution:
    action_per_frame = int(config["lingbotva"].get("action_per_frame", 16))
    temporal_compression = int(config["lingbotva"].get("vae_temporal_compression", 4))
    observation_stride = action_per_frame // temporal_compression
    if len(robot_actions) != action_per_frame:
        raise RuntimeError(
            f"Predictive overlap expects one {action_per_frame}-step frame, got {len(robot_actions)}"
        )

    started_at = time.monotonic()
    executed_actions = robot.send_action_chunk(robot_actions, source_hz=playback_fps)
    if len(executed_actions) != len(robot_actions):
        raise RuntimeError(
            "Robot returned a different number of executed actions: "
            f"{len(executed_actions)} != {len(robot_actions)}"
        )

    observations = []
    final_observation = None
    for completed in range(observation_stride, action_per_frame + 1, observation_stride):
        deadline = started_at + completed / playback_fps
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        final_observation = robot.get_observation()
        observations.append(observation_for_policy(config, final_observation))

    if final_observation is None:
        raise RuntimeError("LingBot-VA action frame produced no cache observation")
    return FrameExecution(
        model_actions=model_actions,
        robot_actions=robot_actions,
        executed_actions=executed_actions,
        observations=observations,
        final_observation=final_observation,
        elapsed_s=time.monotonic() - started_at,
    )


def update_cache_and_infer(
    config: dict,
    policy,
    frame: FrameExecution,
    *,
    cache_prefix: np.ndarray | None = None,
) -> ModelCycle:
    task = config["dataset"]["single_task"]
    executed_model_actions = actual_robot_actions_to_model(config, frame.executed_actions)
    cache_actions = executed_model_actions
    if cache_prefix is not None and len(cache_prefix):
        cache_actions = np.concatenate((cache_prefix, executed_model_actions), axis=0)

    cache_started = time.monotonic()
    policy.infer(
        {
            "compute_kv_cache": True,
            "obs": frame.observations,
            "actions": cache_actions,
            "prompt": task,
        }
    )
    cache_s = time.monotonic() - cache_started

    inference_started = time.monotonic()
    result = policy.infer(observation_for_policy(config, frame.final_observation))
    inference_s = time.monotonic() - inference_started
    model_actions, _, robot_actions = model_actions_to_robot(
        config,
        result,
        first_chunk=False,
    )
    return ModelCycle(
        model_actions=model_actions,
        robot_actions=robot_actions,
        cache_s=cache_s,
        inference_s=inference_s,
        elapsed_s=cache_s + inference_s,
    )


def adaptive_playback_fps(
    options: StutterOptions,
    action_per_frame: int,
    latency_ema_s: float,
) -> float:
    if not options.adaptive_playback:
        return options.playback_fps
    target = action_per_frame / max(
        latency_ema_s * options.latency_safety_factor,
        1e-6,
    )
    return float(np.clip(target, options.min_playback_fps, options.max_playback_fps))


def log_timing_summary(mode: str, timings: list[dict[str, float]]) -> None:
    if not timings:
        return
    inference = np.asarray([item["inference_s"] for item in timings])
    cache = np.asarray([item["cache_s"] for item in timings])
    model_cycle = np.asarray([item["model_cycle_s"] for item in timings])
    gaps = np.asarray([item["gap_s"] for item in timings])
    action = np.asarray([item["action_s"] for item in timings])
    duty = action.sum() / max(action.sum() + gaps.sum(), 1e-6)
    LOG.info(
        "%s summary samples=%d inference_mean=%.3fs cache_mean=%.3fs "
        "model_cycle_mean=%.3fs gap_mean=%.3fs gap_p95=%.3fs motion_duty=%.1f%%",
        mode,
        len(timings),
        float(inference.mean()),
        float(cache.mean()),
        float(model_cycle.mean()),
        float(gaps.mean()),
        float(np.percentile(gaps, 95)),
        duty * 100.0,
    )


def run_predictive_overlap(
    config: dict,
    robot: MarvainM6HttpRobot,
    policy,
    initial_observation: dict,
    should_stop: Callable[[], bool],
) -> None:
    options = StutterOptions.from_config(config)
    runtime = config["lingbotva"]
    action_per_frame = int(runtime.get("action_per_frame", 16))
    duration = float(config["inference"].get("duration", 0.0))
    max_steps = int(config["inference"].get("max_steps", 10000))
    started_at = time.monotonic()
    frame_index = 0
    timings: list[dict[str, float]] = []

    def within_limits() -> bool:
        if should_stop() or frame_index >= max_steps:
            return False
        return duration <= 0.0 or time.monotonic() - started_at < duration

    inference_started = time.monotonic()
    initial_result = policy.infer(observation_for_policy(config, initial_observation))
    initial_inference_s = time.monotonic() - inference_started
    initial_model, cache_prefix, initial_robot = model_actions_to_robot(
        config,
        initial_result,
        first_chunk=True,
    )
    initial_frames = split_action_frames(config, initial_model, initial_robot)
    if len(initial_frames) != 1:
        raise RuntimeError(
            f"LingBot-VA cold start must expose exactly one executable frame, got {len(initial_frames)}"
        )
    playback_fps = options.playback_fps
    LOG.info(
        "predictive_overlap cold_start inference=%.3fs playback_fps=%.2f",
        initial_inference_s,
        playback_fps,
    )
    current_frame = execute_action_frame(
        config,
        robot,
        *initial_frames[0],
        playback_fps,
    )
    frame_index += 1
    if not within_limits():
        return

    bootstrap = update_cache_and_infer(
        config,
        policy,
        current_frame,
        cache_prefix=cache_prefix,
    )
    latency_ema_s = bootstrap.elapsed_s
    playback_fps = adaptive_playback_fps(options, action_per_frame, latency_ema_s)
    planned_frames = split_action_frames(config, bootstrap.model_actions, bootstrap.robot_actions)
    if len(planned_frames) != 2:
        raise RuntimeError(
            f"predictive_overlap requires two predicted frames, got {len(planned_frames)}"
        )
    LOG.info(
        "predictive_overlap bootstrap cache=%.3fs inference=%.3fs model_cycle=%.3fs "
        "adaptive_playback_fps=%.2f",
        bootstrap.cache_s,
        bootstrap.inference_s,
        bootstrap.elapsed_s,
        playback_fps,
    )
    if not within_limits():
        return

    current_frame = execute_action_frame(
        config,
        robot,
        *planned_frames[0],
        playback_fps,
    )
    frame_index += 1
    pending_frame = planned_frames[1]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbotva-model") as executor:
        while within_limits():
            future = executor.submit(update_cache_and_infer, config, policy, current_frame)
            executed_frame = execute_action_frame(
                config,
                robot,
                *pending_frame,
                playback_fps,
            )
            frame_index += 1
            gap_started = time.monotonic()
            model_cycle = future.result()
            gap_s = time.monotonic() - gap_started

            latency_ema_s = (
                options.latency_ema_alpha * model_cycle.elapsed_s
                + (1.0 - options.latency_ema_alpha) * latency_ema_s
            )
            next_playback_fps = adaptive_playback_fps(
                options,
                action_per_frame,
                latency_ema_s,
            )
            timing = {
                "inference_s": model_cycle.inference_s,
                "cache_s": model_cycle.cache_s,
                "model_cycle_s": model_cycle.elapsed_s,
                "action_s": executed_frame.elapsed_s,
                "gap_s": gap_s,
            }
            timings.append(timing)
            if frame_index == 3 or frame_index % options.log_every_n_frames == 0:
                LOG.info(
                    "overlap frame=%d action=%.3fs cache=%.3fs inference=%.3fs "
                    "model_cycle=%.3fs gap=%.3fs playback_fps=%.2f->%.2f",
                    frame_index,
                    executed_frame.elapsed_s,
                    model_cycle.cache_s,
                    model_cycle.inference_s,
                    model_cycle.elapsed_s,
                    gap_s,
                    playback_fps,
                    next_playback_fps,
                )

            if not within_limits():
                break
            next_frames = split_action_frames(
                config,
                model_cycle.model_actions,
                model_cycle.robot_actions,
            )
            if len(next_frames) != 2:
                raise RuntimeError(
                    f"predictive_overlap requires two predicted frames, got {len(next_frames)}"
                )
            # Frame 0 overlaps the frame that just executed. Frame 1 is the new lookahead.
            pending_frame = next_frames[1]
            current_frame = executed_frame
            playback_fps = next_playback_fps

    log_timing_summary("predictive_overlap", timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = load_config(args.config)
    if args.duration is not None:
        config["inference"]["duration"] = args.duration
    validate_config(config)
    robot = build_robot(config)
    if args.validate_only:
        LOG.info("LingBot-VA robot client configuration is valid; no connection was made")
        return 0

    runtime = config["lingbotva"]
    host = args.policy_host or runtime.get("server_host", "127.0.0.1")
    port = args.policy_port or int(runtime.get("server_port", 8002))
    LOG.info("Connecting to LingBot-VA policy server at ws://%s:%d", host, port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    LOG.info("LingBot-VA server metadata: %s", policy.get_server_metadata())

    task = config["dataset"]["single_task"]
    policy.infer({"reset": True, "prompt": task})
    LOG.info("LingBot-VA episode cache reset with the configured task prompt")

    stop_requested = False

    def request_stop(signum: int, _frame) -> None:
        nonlocal stop_requested
        LOG.info("Received signal %d; stopping after the current action block", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    fps = float(config["inference"].get("fps", 30.0))
    duration = float(config["inference"].get("duration", 0.0))
    max_steps = int(config["inference"].get("max_steps", 10000))
    action_per_frame = int(runtime.get("action_per_frame", 16))
    temporal_compression = int(runtime.get("vae_temporal_compression", 4))
    observation_stride = action_per_frame // temporal_compression
    stutter = StutterOptions.from_config(config)
    started_at = time.monotonic()
    iteration = 0
    first_chunk = True

    LOG.info("Connecting to the robot only after LingBot-VA service readiness and reset")
    robot.connect()
    try:
        current_obs = robot.get_observation()
        LOG.info(
            "LingBot-VA stutter optimization: enabled=%s mode=%s",
            stutter.enabled,
            stutter.mode,
        )
        if stutter.mode == "predictive_overlap":
            run_predictive_overlap(
                config,
                robot,
                policy,
                current_obs,
                lambda: stop_requested,
            )
            return 0
        while not stop_requested and iteration < max_steps:
            if duration > 0 and time.monotonic() - started_at >= duration:
                break

            inference_started = time.monotonic()
            result = policy.infer(observation_for_policy(config, current_obs))
            inference_s = time.monotonic() - inference_started
            model_actions, cache_prefix, robot_actions = model_actions_to_robot(
                config,
                result,
                first_chunk=first_chunk,
            )

            executed_actions = []
            cache_observations = []
            for offset in range(0, len(robot_actions), observation_stride):
                block = robot_actions[offset : offset + observation_stride]
                block_started = time.monotonic()
                actual_block = robot.send_action_chunk(block)
                if len(actual_block) != len(block):
                    raise RuntimeError(
                        "Robot returned a different number of executed actions: "
                        f"{len(actual_block)} != {len(block)}"
                    )
                executed_actions.extend(actual_block)
                remaining = len(block) / fps - (time.monotonic() - block_started)
                if remaining > 0:
                    time.sleep(remaining)
                current_obs = robot.get_observation()
                cache_observations.append(observation_for_policy(config, current_obs))
                if stop_requested:
                    break

            if len(executed_actions) != len(robot_actions):
                LOG.info(
                    "Stopping before KV-cache update because only %d/%d actions executed",
                    len(executed_actions),
                    len(robot_actions),
                )
                break

            executed_model_actions = actual_robot_actions_to_model(config, executed_actions)
            cache_actions = np.concatenate(
                (cache_prefix, executed_model_actions),
                axis=0,
            )
            expected_observations = len(executed_model_actions) // observation_stride
            if len(cache_observations) != expected_observations:
                raise RuntimeError(
                    "LingBot-VA KV-cache observation count mismatch: "
                    f"{len(cache_observations)} != {expected_observations}"
                )
            policy.infer(
                {
                    "compute_kv_cache": True,
                    "obs": cache_observations,
                    "actions": cache_actions,
                    "prompt": task,
                }
            )
            first_chunk = False

            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                LOG.info(
                    "chunk=%d steps=%d inference=%.3fs model_deg=[%.2f, %.2f] robot_rad=[%.3f, %.3f]",
                    iteration,
                    len(executed_actions),
                    inference_s,
                    float(model_actions[:, [*range(7), *range(8, 15)]].min()),
                    float(model_actions[:, [*range(7), *range(8, 15)]].max()),
                    min(float(action[f"{name}.pos"]) for action in executed_actions for name in config["robot"]["joint_names"][:14]),
                    max(float(action[f"{name}.pos"]) for action in executed_actions for name in config["robot"]["joint_names"][:14]),
                )
    finally:
        try:
            robot.disconnect()
            LOG.info("Robot disconnected cleanly")
        finally:
            ws = getattr(policy, "_ws", None)
            if ws is not None:
                ws.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

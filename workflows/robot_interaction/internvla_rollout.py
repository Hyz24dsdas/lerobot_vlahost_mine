#!/usr/bin/env python3
"""Run InternVLA-A1.5 against the existing Marvain HTTP robot driver."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


LOG = logging.getLogger("internvla_rollout")
ACTION = "action"
OBS_STATE = "observation.state"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def import_robot_driver(vlahost_repo: Path):
    """Expose the VLAHost robot package alongside InternVLA's slim LeRobot fork."""
    import lerobot
    import lerobot.utils

    current_lerobot = str(vlahost_repo / "src" / "lerobot")
    current_utils = str(vlahost_repo / "src" / "lerobot" / "utils")
    if current_lerobot not in lerobot.__path__:
        lerobot.__path__.append(current_lerobot)
    if current_utils not in lerobot.utils.__path__:
        lerobot.utils.__path__.append(current_utils)

    from lerobot.robots.marvain_m6_http.config_marvain_m6_http import (
        HttpCameraConfig,
        MarvainM6HttpRobotConfig,
    )
    from lerobot.robots.marvain_m6_http.marvain_m6_http import MarvainM6HttpRobot

    return HttpCameraConfig, MarvainM6HttpRobotConfig, MarvainM6HttpRobot


def build_robot(config: dict):
    vlahost_repo = Path(config["internvla"]["vlahost_repo"]).expanduser().resolve()
    HttpCameraConfig, RobotConfig, Robot = import_robot_driver(vlahost_repo)
    robot_cfg = config["robot"]
    cameras = {}
    for name, values in robot_cfg.get("cameras", {}).items():
        values = dict(values)
        camera_type = values.pop("type", "http")
        if camera_type != "http":
            raise ValueError(f"Camera {name!r} must use type=http, got {camera_type!r}")
        cameras[name] = HttpCameraConfig(**values)

    safety_path = robot_cfg.get("safety_stats_path")
    return Robot(
        RobotConfig(
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
    )


def load_stats(checkpoint: Path, robot_type: str) -> dict:
    stats_path = checkpoint / "stats.json"
    with stats_path.open("r", encoding="utf-8") as handle:
        all_stats = json.load(handle)
    stats = all_stats.get(robot_type, all_stats)
    for key in (OBS_STATE, ACTION):
        if key not in stats:
            raise ValueError(f"{stats_path} has no {robot_type!r}/{key!r} statistics")
        for field in ("mean", "std", "min", "max"):
            values = np.asarray(stats[key][field], dtype=np.float32)
            if values.shape != (16,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key}/{field} statistics: shape={values.shape}")
    return stats


def load_policy(config: dict, checkpoint: Path):
    # The fine-tuned checkpoint is complete, so the local Qwen directory only
    # needs architecture and tokenizer assets during deployment.
    os.environ.setdefault("INTERNVLA_INIT_FROM_CONFIG", "1")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy
    from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
        InternVLAA15ChatProcessorTransformFn,
    )

    runtime = config["internvla"]
    qwen_path = str(Path(runtime["qwen_path"]).expanduser().resolve())
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, InternVLAA15Config):
        raise TypeError(f"Expected InternVLAA15Config, got {type(policy_config).__name__}")

    policy_config.device = config["policy"].get("device", "cuda:0")
    policy_config.vlm_model_name_or_path = qwen_path
    policy_config.inference_backend = runtime.get("inference_backend", "optimized")
    policy_config.action_loss_only = bool(runtime.get("action_loss_only", True))
    policy_config.use_sdpa = bool(runtime.get("use_sdpa", True))
    policy_config.gradient_checkpointing = False
    policy_config.compile_model = bool(runtime.get("compile_model", False))
    policy_config.n_action_steps = policy_config.chunk_size

    if policy_config.inference_backend != "optimized" or not policy_config.action_loss_only:
        raise ValueError("Real-robot InternVLA deployment requires optimized action-only inference")

    policy = InternVLAA15Policy.from_pretrained(checkpoint, config=policy_config)
    policy.to(device=torch.device(policy_config.device), dtype=torch.bfloat16)
    policy.eval()
    processor = InternVLAA15ChatProcessorTransformFn(
        pretrained_model_name_or_path=qwen_path,
        max_length=int(runtime.get("max_prompt_length", 650)),
        tokenize_state=True,
        max_state_dim=policy_config.max_state_dim,
        use_fast_action_tokens=True,
        mode="eval",
        action_mode="joint",
    )
    return policy, processor, policy_config


def validate_config(config: dict, checkpoint: Path) -> None:
    if len(config["robot"].get("joint_names", [])) != 16:
        raise ValueError("InternVLA deployment requires exactly 16 joint_names")
    mapping = config["internvla"].get("observation", {})
    required = {"image0", "image1", "image2"}
    if set(mapping) != required:
        raise ValueError(f"internvla.observation must define exactly {sorted(required)}")
    missing_cameras = set(mapping.values()) - set(config["robot"].get("cameras", {}))
    if missing_cameras:
        raise ValueError(f"InternVLA camera mapping references unknown cameras: {missing_cameras}")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    n_action_steps = int(config["inference"].get("n_action_steps", 30))
    if not 1 <= n_action_steps <= 50:
        raise ValueError("inference.n_action_steps must be in [1, 50]")


def observation_batch(config: dict, processor, stats: dict, robot_obs: dict, device: torch.device):
    from lerobot.transforms.core import resize_with_pad

    image_mapping = config["internvla"]["observation"]
    missing = [name for name in image_mapping.values() if name not in robot_obs]
    if missing:
        raise RuntimeError(f"Robot observation is missing InternVLA cameras: {missing}")

    names = config["robot"]["joint_names"]
    state = np.asarray([robot_obs[f"{name}.pos"] for name in names], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid robot state: shape={state.shape}")
    state_stats = stats[OBS_STATE]
    mean = np.asarray(state_stats["mean"], dtype=np.float32)
    std = np.asarray(state_stats["std"], dtype=np.float32)
    normalized_state = (state - mean) / (std + 1e-6)

    sample = {
        OBS_STATE: torch.from_numpy(normalized_state),
        "task": config["dataset"]["single_task"],
    }
    height, width = config["internvla"].get("image_resolution", [224, 224])
    for index in range(3):
        source = image_mapping[f"image{index}"]
        image = np.asarray(robot_obs[source])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise RuntimeError(f"Invalid camera {source}: shape={image.shape}, dtype={image.dtype}")
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        key = f"observation.images.image{index}"
        sample[key] = resize_with_pad(tensor, int(height), int(width), "bilinear")
        sample[f"{key}_mask"] = torch.tensor(True)

    sample = processor(sample)
    keys = (
        OBS_STATE,
        "observation.input_ids",
        "observation.attention_mask",
        "observation.pixel_values",
        "observation.image_grid_thw",
        "observation.fast_token_mask",
    )
    batch = {}
    for key in keys:
        value = sample[key]
        value = value.unsqueeze(0).to(device=device)
        if value.dtype not in (torch.int64, torch.int32, torch.bool):
            value = value.to(dtype=torch.bfloat16)
        batch[key] = value
    return batch


def actions_for_robot(config: dict, stats: dict, prediction: torch.Tensor) -> list[dict[str, float]]:
    actions = prediction[0, :, :16].detach().float().cpu().numpy()
    action_stats = stats[ACTION]
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    actions = actions * (std + 1e-6) + mean
    if not np.isfinite(actions).all():
        raise RuntimeError("InternVLA returned NaN or Inf actions")

    n_steps = min(int(config["inference"].get("n_action_steps", 30)), len(actions))
    actions = actions[:n_steps]
    gripper_min = float(config["internvla"].get("gripper_min", 0.0))
    gripper_max = float(config["internvla"].get("gripper_max", 1.0))
    actions[:, 14:16] = np.clip(actions[:, 14:16], gripper_min, gripper_max)
    names = config["robot"]["joint_names"]
    return [
        {f"{name}.pos": float(value) for name, value in zip(names, row, strict=True)}
        for row in actions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    validate_config(config, checkpoint)
    stats = load_stats(checkpoint, config["internvla"].get("stats_robot_type", "marvin"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVLA real-robot deployment")
    device = torch.device(config["policy"].get("device", "cuda:0"))
    LOG.info("Loading InternVLA checkpoint in optimized action-only mode: %s", checkpoint)
    policy, processor, policy_config = load_policy(config, checkpoint)
    LOG.info(
        "InternVLA ready: backend=%s action_only=%s chunk=%d device=%s",
        policy_config.inference_backend,
        policy_config.action_loss_only,
        policy_config.chunk_size,
        device,
    )
    if args.validate_only:
        return 0

    stop_requested = False

    def request_stop(signum: int, _frame) -> None:
        nonlocal stop_requested
        LOG.info("Received signal %d; stopping safely", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(str(os.getpid()), encoding="ascii")
        LOG.info("Model-ready signal written to %s", args.ready_file)
    if args.start_file:
        LOG.info("Waiting for deployment start signal: %s", args.start_file)
        while not args.start_file.exists() and not stop_requested:
            time.sleep(0.1)
    if stop_requested:
        return 130

    robot = build_robot(config)
    fps = float(config["inference"].get("fps", 30.0))
    duration = float(config["inference"].get("duration", 0.0))
    max_steps = int(config["inference"].get("max_steps", 10000))
    started_at = time.monotonic()
    iteration = 0

    LOG.info("Connecting to the robot only after model readiness and wrapper start signal")
    robot.connect()
    try:
        while not stop_requested and iteration < max_steps:
            if duration > 0 and time.monotonic() - started_at >= duration:
                break
            cycle_started = time.monotonic()
            batch = observation_batch(config, processor, stats, robot.get_observation(), device)
            infer_started = time.monotonic()
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(batch)
            inference_s = time.monotonic() - infer_started
            actions = actions_for_robot(config, stats, prediction)
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
            remaining = len(actions) / fps - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        robot.disconnect()
        LOG.info("Robot disconnected cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Serve an external InternVLA-A1.5 checkpoint over loopback HTTP."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml


LOG = logging.getLogger("internvla_policy_server")
ACTION = "action"
OBS_STATE = "observation.state"
MAX_REQUEST_BYTES = 32 * 1024 * 1024


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


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


def validate_config(config: dict, checkpoint: Path) -> None:
    if len(config["robot"].get("joint_names", [])) != 16:
        raise ValueError("InternVLA deployment requires exactly 16 joint_names")
    mapping = config["internvla"].get("observation", {})
    required = {"image0", "image1", "image2"}
    if set(mapping) != required:
        raise ValueError(f"internvla.observation must define exactly {sorted(required)}")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    n_action_steps = int(config["inference"].get("n_action_steps", 30))
    if not 1 <= n_action_steps <= 50:
        raise ValueError("inference.n_action_steps must be in [1, 50]")
    task = config["dataset"].get("single_task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("dataset.single_task must be a non-empty string")


def load_policy(config: dict, checkpoint: Path):
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


class PolicyRuntime:
    def __init__(self, config: dict, checkpoint: Path):
        self.config = config
        self.checkpoint = checkpoint
        self.stats = load_stats(
            checkpoint,
            config["internvla"].get("stats_robot_type", "marvin"),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for InternVLA real-robot deployment")
        self.device = torch.device(config["policy"].get("device", "cuda:0"))
        LOG.info("Loading InternVLA checkpoint: %s", checkpoint)
        self.policy, self.processor, self.policy_config = load_policy(config, checkpoint)
        LOG.info(
            "InternVLA loaded: backend=%s action_only=%s chunk=%d device=%s",
            self.policy_config.inference_backend,
            self.policy_config.action_loss_only,
            self.policy_config.chunk_size,
            self.device,
        )

    def _batch(self, state: np.ndarray, images: list[np.ndarray], prompt: str) -> dict:
        from lerobot.transforms.core import resize_with_pad

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid state: shape={state.shape}")
        if prompt != self.config["dataset"]["single_task"]:
            raise ValueError("Request prompt does not match dataset.single_task")

        state_stats = self.stats[OBS_STATE]
        mean = np.asarray(state_stats["mean"], dtype=np.float32)
        std = np.asarray(state_stats["std"], dtype=np.float32)
        sample = {
            OBS_STATE: torch.from_numpy((state - mean) / (std + 1e-6)),
            "task": prompt,
        }
        height, width = self.config["internvla"].get("image_resolution", [224, 224])
        for index, image in enumerate(images):
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Invalid image{index}: shape={image.shape}, dtype={image.dtype}")
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
            key = f"observation.images.image{index}"
            sample[key] = resize_with_pad(tensor, int(height), int(width), "bilinear")
            sample[f"{key}_mask"] = torch.tensor(True)

        sample = self.processor(sample)
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
            value = sample[key].unsqueeze(0).to(device=self.device)
            if value.dtype not in (torch.int64, torch.int32, torch.bool):
                value = value.to(dtype=torch.bfloat16)
            batch[key] = value
        return batch

    def infer(self, state: np.ndarray, images: list[np.ndarray], prompt: str) -> np.ndarray:
        if len(images) != 3:
            raise ValueError(f"Expected three images, got {len(images)}")
        batch = self._batch(state, images, prompt)
        started = time.monotonic()
        with torch.inference_mode():
            prediction = self.policy.predict_action_chunk(batch)
        if prediction.ndim != 3 or prediction.shape[0] != 1:
            raise RuntimeError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
        if prediction.shape[1] != self.policy_config.chunk_size or prediction.shape[2] < 16:
            raise RuntimeError(f"Unexpected action dimensions: {tuple(prediction.shape)}")
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("InternVLA returned NaN or Inf")

        actions = prediction[0, :, :16].detach().float().cpu().numpy()
        action_stats = self.stats[ACTION]
        mean = np.asarray(action_stats["mean"], dtype=np.float32)
        std = np.asarray(action_stats["std"], dtype=np.float32)
        actions = actions * (std + 1e-6) + mean
        n_steps = min(int(self.config["inference"].get("n_action_steps", 30)), len(actions))
        actions = actions[:n_steps]
        actions[:, 14:16] = np.clip(
            actions[:, 14:16],
            float(self.config["internvla"].get("gripper_min", 0.0)),
            float(self.config["internvla"].get("gripper_max", 1.0)),
        )
        if not np.isfinite(actions).all():
            raise RuntimeError("Denormalized InternVLA actions contain NaN or Inf")
        LOG.info(
            "Inference completed: %.3fs output=%s range=[%.4f, %.4f]",
            time.monotonic() - started,
            actions.shape,
            float(actions.min()),
            float(actions.max()),
        )
        return actions.astype(np.float32, copy=False)

    def smoke_test(self) -> None:
        state = np.asarray(self.stats[OBS_STATE]["mean"], dtype=np.float32)
        height, width = self.config["internvla"].get("image_resolution", [224, 224])
        images = [
            np.full((int(height), int(width), 3), 127, dtype=np.uint8)
            for _ in range(3)
        ]
        actions = self.infer(state, images, self.config["dataset"]["single_task"])
        if actions.shape != (int(self.config["inference"].get("n_action_steps", 30)), 16):
            raise RuntimeError(f"Unexpected smoke-test actions: {actions.shape}")
        LOG.info("Offline smoke inference passed: actions=%s", actions.shape)


def make_handler(runtime: PolicyRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(
                200,
                {
                    "status": "ready",
                    "model": "internvla_a1_5",
                    "checkpoint": str(runtime.checkpoint),
                    "chunk_size": runtime.policy_config.chunk_size,
                    "n_action_steps": int(runtime.config["inference"].get("n_action_steps", 30)),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/infer":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"Invalid request size: {length}")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request body")
                with np.load(io.BytesIO(body), allow_pickle=False) as payload:
                    required = {"state", "image0", "image1", "image2", "prompt"}
                    if set(payload.files) != required:
                        raise ValueError(f"Unexpected request fields: {sorted(payload.files)}")
                    state = payload["state"].copy()
                    images = [payload[f"image{index}"].copy() for index in range(3)]
                    prompt = str(payload["prompt"].item())
                actions = runtime.infer(state, images, prompt)
                output = io.BytesIO()
                np.savez(output, actions=actions)
                self._send(200, "application/x-npz", output.getvalue())
            except Exception as exc:
                LOG.exception("InternVLA inference request failed")
                self._send_json(500, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            LOG.debug("HTTP %s", format % args)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("InternVLA policy server must bind to loopback")
    config = load_config(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    validate_config(config, checkpoint)
    runtime = PolicyRuntime(config, checkpoint)
    runtime.smoke_test()
    if args.validate_only:
        return 0

    server = HTTPServer((args.host, args.port), make_handler(runtime))

    def request_stop(signum: int, _frame) -> None:
        LOG.info("Received signal %d; stopping policy server", signum)
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    LOG.info("InternVLA policy server ready at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

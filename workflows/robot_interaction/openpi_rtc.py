"""Runtime scheduling for native OpenPI Real-Time Chunking deployment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping


_PREFIX_ELIGIBLE_EXECUTIONS = {"active", "resume_pending"}
_SCHEDULES = {"LINEAR", "EXP", "ONES", "ZEROS"}


@dataclass(frozen=True)
class RTCRuntimeConfig:
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "EXP"
    queue_threshold: int = 30
    initial_inference_delay_steps: int = 3
    latency_margin_steps: int = 0
    latency_window: int = 20
    max_chunk_steps: int = 50
    min_send_steps: int = 10

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RTCRuntimeConfig":
        values = values or {}
        config = cls(
            execution_horizon=int(values.get("execution_horizon", 10)),
            max_guidance_weight=float(values.get("max_guidance_weight", 10.0)),
            prefix_attention_schedule=str(values.get("prefix_attention_schedule", "EXP")).upper(),
            queue_threshold=int(values.get("queue_threshold", 30)),
            initial_inference_delay_steps=int(values.get("initial_inference_delay_steps", 3)),
            latency_margin_steps=int(values.get("latency_margin_steps", 0)),
            latency_window=int(values.get("latency_window", 20)),
            max_chunk_steps=int(values.get("max_chunk_steps", 50)),
            min_send_steps=int(values.get("min_send_steps", 10)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.execution_horizon <= 50:
            raise ValueError("inference.rtc.execution_horizon must be in [1, 50]")
        if self.max_guidance_weight < 0.0:
            raise ValueError("inference.rtc.max_guidance_weight must be non-negative")
        if self.prefix_attention_schedule not in _SCHEDULES:
            raise ValueError(
                "inference.rtc.prefix_attention_schedule must be one of "
                f"{sorted(_SCHEDULES)}"
            )
        if not 1 <= self.max_chunk_steps <= 50:
            raise ValueError("inference.rtc.max_chunk_steps must be in [1, 50]")
        if not 0 <= self.queue_threshold < self.max_chunk_steps:
            raise ValueError(
                "inference.rtc.queue_threshold must be in [0, max_chunk_steps)"
            )
        if not 0 <= self.initial_inference_delay_steps < self.max_chunk_steps:
            raise ValueError(
                "inference.rtc.initial_inference_delay_steps must be in "
                "[0, max_chunk_steps)"
            )
        if self.latency_margin_steps < 0:
            raise ValueError("inference.rtc.latency_margin_steps must be non-negative")
        if self.latency_window < 1:
            raise ValueError("inference.rtc.latency_window must be positive")
        if not 1 <= self.min_send_steps <= self.max_chunk_steps:
            raise ValueError(
                "inference.rtc.min_send_steps must be in [1, max_chunk_steps]"
            )


@dataclass(frozen=True)
class RTCRequestContext:
    payload: dict[str, Any]
    had_previous: bool
    prefix_start_step: int
    estimated_delay_steps: int


@dataclass(frozen=True)
class RTCActionWindow:
    start_step: int
    end_step: int

    @property
    def count(self) -> int:
        return self.end_step - self.start_step


class RTCScheduler:
    """Track action age and align asynchronous OpenPI chunks to robot time."""

    def __init__(self, config: RTCRuntimeConfig, fps: float):
        if fps <= 0.0:
            raise ValueError("RTC fps must be positive")
        self.config = config
        self.fps = float(fps)
        self._latencies: deque[float] = deque(maxlen=config.latency_window)
        self._server_chunk_id: int | None = None
        self._output_offset = 0
        self._sent_steps = 0
        self._sent_at: float | None = None
        self._prefix_eligible = False
        self._has_dispatched = False

    def remaining_steps(self, now: float) -> int:
        if self._sent_at is None:
            return 0
        elapsed_steps = max(0, math.floor((now - self._sent_at) * self.fps))
        return max(0, self._sent_steps - elapsed_steps)

    def seconds_until_inference(self, now: float) -> float:
        remaining = self.remaining_steps(now)
        if remaining <= self.config.queue_threshold:
            return 0.0
        return (remaining - self.config.queue_threshold) / self.fps

    def estimated_delay_steps(self) -> int:
        if self._latencies:
            delay = math.ceil(max(self._latencies) * self.fps)
            delay += self.config.latency_margin_steps
        else:
            delay = self.config.initial_inference_delay_steps
        return max(0, min(delay, self.config.max_chunk_steps - 1))

    def build_request(self, now: float, *, reset: bool = False) -> RTCRequestContext:
        executed_steps = self._sent_steps - self.remaining_steps(now)
        had_previous = (
            not reset
            and self._prefix_eligible
            and self._server_chunk_id is not None
            and self._sent_at is not None
        )
        prefix_start = self._output_offset + executed_steps if had_previous else 0
        estimated_delay = self.estimated_delay_steps()
        payload = {
            "enabled": True,
            "reset": not had_previous,
            "previous_chunk_id": self._server_chunk_id if had_previous else -1,
            "prefix_start_step": prefix_start,
            "inference_delay_steps": estimated_delay,
            "execution_horizon": self.config.execution_horizon,
            "max_guidance_weight": self.config.max_guidance_weight,
            "prefix_attention_schedule": self.config.prefix_attention_schedule,
        }
        return RTCRequestContext(
            payload=payload,
            had_previous=had_previous,
            prefix_start_step=prefix_start,
            estimated_delay_steps=estimated_delay,
        )

    def action_window(
        self,
        action_count: int,
        round_trip_s: float,
        request: RTCRequestContext,
    ) -> RTCActionWindow:
        if action_count < 1:
            raise RuntimeError("OpenPI RTC returned an empty action chunk")
        discard = math.ceil(max(0.0, round_trip_s) * self.fps) if request.had_previous else 0
        discard = min(discard, action_count)
        end = min(action_count, discard + self.config.max_chunk_steps)
        if end - discard < self.config.min_send_steps:
            raise RuntimeError(
                "OpenPI RTC inference left too few current actions: "
                f"returned={action_count}, stale={discard}, usable={end - discard}, "
                f"required={self.config.min_send_steps}"
            )
        return RTCActionWindow(discard, end)

    def record_dispatch(
        self,
        *,
        rtc_response: Mapping[str, Any],
        request: RTCRequestContext,
        round_trip_s: float,
        window: RTCActionWindow,
        execution: str,
        sent_at: float,
    ) -> None:
        if "chunk_id" not in rtc_response:
            raise RuntimeError("OpenPI RTC response is missing chunk_id")
        if self._has_dispatched:
            self._latencies.append(max(0.0, float(round_trip_s)))
        self._server_chunk_id = int(rtc_response["chunk_id"])
        self._output_offset = window.start_step
        self._sent_steps = window.count
        self._sent_at = float(sent_at)
        self._prefix_eligible = execution in _PREFIX_ELIGIBLE_EXECUTIONS
        self._has_dispatched = True

    def reset_prefix(self) -> None:
        self._prefix_eligible = False


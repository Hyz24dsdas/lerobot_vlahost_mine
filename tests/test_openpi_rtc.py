from pathlib import Path
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows" / "robot_interaction"
sys.path.insert(0, str(WORKFLOW_DIR))

from openpi_rtc import RTCActionWindow, RTCRuntimeConfig, RTCScheduler  # noqa: E402


def test_rtc_config_rejects_invalid_queue_threshold():
    with pytest.raises(ValueError, match="queue_threshold"):
        RTCRuntimeConfig.from_mapping({"max_chunk_steps": 50, "queue_threshold": 50})


def test_first_chunk_does_not_discard_jit_latency():
    scheduler = RTCScheduler(RTCRuntimeConfig(), fps=30.0)
    request = scheduler.build_request(100.0)
    window = scheduler.action_window(50, 10.0, request)
    assert request.had_previous is False
    assert request.payload["reset"] is True
    assert window == RTCActionWindow(0, 50)


def test_rtc_aligns_second_chunk_and_waits_for_queue_threshold():
    scheduler = RTCScheduler(RTCRuntimeConfig(), fps=30.0)
    first_request = scheduler.build_request(100.0)
    scheduler.record_dispatch(
        rtc_response={"chunk_id": 1, "prefix_applied": False},
        request=first_request,
        round_trip_s=10.0,
        window=RTCActionWindow(0, 50),
        execution="active",
        sent_at=100.0,
    )

    assert scheduler.seconds_until_inference(100.0) == pytest.approx(20 / 30)
    second_started = 100.0 + 20 / 30
    second_request = scheduler.build_request(second_started)
    assert second_request.had_previous is True
    assert second_request.prefix_start_step == 20
    assert second_request.estimated_delay_steps == 3

    window = scheduler.action_window(50, 0.087, second_request)
    assert window == RTCActionWindow(3, 50)
    scheduler.record_dispatch(
        rtc_response={"chunk_id": 2, "prefix_applied": True},
        request=second_request,
        round_trip_s=0.087,
        window=window,
        execution="active",
        sent_at=second_started + 0.087,
    )
    assert scheduler.seconds_until_inference(second_started + 0.087) == pytest.approx(17 / 30)
    assert scheduler.estimated_delay_steps() == 3


def test_shadow_chunk_is_not_used_as_rtc_prefix():
    scheduler = RTCScheduler(RTCRuntimeConfig(), fps=30.0)
    request = scheduler.build_request(0.0)
    scheduler.record_dispatch(
        rtc_response={"chunk_id": 1},
        request=request,
        round_trip_s=0.1,
        window=RTCActionWindow(0, 50),
        execution="shadow",
        sent_at=0.1,
    )
    next_request = scheduler.build_request(1.0)
    assert next_request.had_previous is False
    assert next_request.payload["reset"] is True

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for raw gripper-value passthrough in the Marvain M6 HTTP driver."""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.marvain_m6_http import MarvainM6HttpRobot, MarvainM6HttpRobotConfig


MODULE = "lerobot.robots.marvain_m6_http.marvain_m6_http"


@pytest.fixture
def http_robot(tmp_path):
    session = MagicMock()
    with patch(f"{MODULE}.requests.Session", return_value=session):
        config = MarvainM6HttpRobotConfig(
            id="raw_gripper_test",
            calibration_dir=tmp_path,
            http_base_url="http://127.0.0.1:8010",
            warn_on_observation_out_of_range=False,
        )
        robot = MarvainM6HttpRobot(config)
    return robot, session


def test_get_observation_preserves_raw_gripper_values(http_robot):
    robot, session = http_robot
    response = MagicMock()
    response.json.return_value = {
        "joint_states": {"positions": [0.0] * 14},
        "gripper_left": {"position": 0.25},
        "gripper_right": {"position": 0.75},
    }
    session.get.return_value = response
    robot._connected = True

    observation = robot.get_observation()

    assert observation["left_gripper.pos"] == pytest.approx(0.25)
    assert observation["right_gripper.pos"] == pytest.approx(0.75)


def test_prepare_action_preserves_raw_gripper_values(http_robot):
    robot, _ = http_robot
    action = {f"{name}.pos": 0.0 for name in robot.config.joint_names}
    action["left_gripper.pos"] = 0.25
    action["right_gripper.pos"] = 0.75

    payload, result = robot._prepare_action(action)

    assert payload["gripper_left"] == pytest.approx(0.25)
    assert payload["gripper_right"] == pytest.approx(0.75)
    assert result["left_gripper.pos"] == pytest.approx(0.25)
    assert result["right_gripper.pos"] == pytest.approx(0.75)

"""Verification checks for kinematics, control units and deterministic trials."""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import robot_xml
from study import GEAR, TORQUE_LIMIT, desired_ee_xyz, simulate


def test_forward_kinematics_against_mujoco():
    model = mujoco.MjModel.from_xml_string(robot_xml())
    data = mujoco.MjData(model)
    q = np.array([0.2, -0.12, 0.18])
    data.qpos[:3] = q
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    np.testing.assert_allclose(data.site_xpos[site], desired_ee_xyz(q), atol=1e-8)


def test_gears_convert_controls_to_joint_torque():
    model = mujoco.MjModel.from_xml_string(robot_xml())
    data = mujoco.MjData(model)
    tau = np.array([1.0, -2.0, 0.8])
    data.ctrl[:] = tau / GEAR
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[:3], tau, rtol=1e-10, atol=1e-10)


def test_paired_replicates_are_deterministic_and_bounded():
    row1, trace1 = simulate("combined", "adaptive_gravity", 3, 12.0, True)
    row2, trace2 = simulate("combined", "adaptive_gravity", 3, 12.0, True)
    assert row1 == row2
    for key in trace1:
        np.testing.assert_array_equal(trace1[key], trace2[key])
    assert np.max(np.abs(trace1["torque"]), axis=0).shape == (3,)
    assert np.all(np.abs(trace1["torque"]) <= TORQUE_LIMIT + 1e-10)
    assert row1["finite"] == 1

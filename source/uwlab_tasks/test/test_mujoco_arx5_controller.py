from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROBOT_XML = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_robot.xml"


def _make_model_and_data():
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "isaac_default")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_controller_resolves_arx5_mujoco_interface_and_train_gains():
    _, model, _ = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")

    assert controller.action_dim == 7
    assert controller.arm_joint_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    assert controller.arm_actuator_names == (
        "joint1_torque",
        "joint2_torque",
        "joint3_torque",
        "joint4_torque",
        "joint5_torque",
        "joint6_torque",
    )
    assert controller.gripper_actuator_names == ("joint7_position", "joint8_position")
    assert controller.position_scale == pytest.approx(0.02)
    assert controller.orientation_scale == pytest.approx(0.2)
    assert controller.motion_stiffness == pytest.approx([200.0, 200.0, 200.0, 3.0, 3.0, 3.0])
    assert controller.motion_damping == pytest.approx(
        2.0 * np.sqrt([200.0, 200.0, 200.0, 3.0, 3.0, 3.0]) * [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]
    )


def test_set_target_scales_policy_action_and_maps_binary_gripper():
    _, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    initial_pose = controller.get_ee_pose(data)

    action = np.array([1.0, -0.5, 0.25, 0.5, 0.0, 1.0, -0.1])
    target = controller.set_target_from_action(data, action)

    assert controller.raw_action == pytest.approx(action)
    assert controller.processed_arm_action == pytest.approx([0.02, -0.01, 0.005, 0.1, 0.0, 0.2])
    assert target.position == pytest.approx(initial_pose.position + np.array([0.02, -0.01, 0.005]))
    assert np.linalg.norm(target.quaternion) == pytest.approx(1.0)
    assert target.gripper == pytest.approx({"joint7": 0.002, "joint8": 0.002})


def test_zero_action_after_reset_writes_zero_arm_torque_and_open_gripper():
    _, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    controller.reset(data)
    output = controller.apply_action(data, np.zeros(7))

    assert output.pose_error == pytest.approx(np.zeros(6), abs=1e-9)
    assert data.ctrl[controller.arm_actuator_ids] == pytest.approx(np.zeros(6), abs=1e-9)
    assert data.ctrl[controller.gripper_actuator_ids] == pytest.approx([0.044, 0.044])


def test_apply_action_writes_finite_clipped_torques_to_mujoco_ctrl():
    _, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    output = controller.apply_action(data, [0.5, -0.25, 0.1, 0.2, -0.1, 0.3, -1.0])

    assert output.torque.shape == (6,)
    assert np.all(np.isfinite(output.torque))
    assert np.max(np.abs(output.torque)) <= 50.0
    assert data.ctrl[controller.arm_actuator_ids] == pytest.approx(output.torque)
    assert data.ctrl[controller.gripper_actuator_ids] == pytest.approx([0.002, 0.002])


def test_apply_target_holds_relative_action_target_across_physics_steps():
    mujoco, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    target = controller.set_target_from_action(data, [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    target_position = target.position.copy()

    for _ in range(12):
        controller.apply_target(data)
        mujoco.mj_step(model, data)

    assert controller.target.position == pytest.approx(target_position)
    assert controller.processed_arm_action == pytest.approx([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert data.ctrl[controller.gripper_actuator_ids] == pytest.approx([0.044, 0.044])


def test_controller_zero_action_rollout_keeps_gripper_state_bounded():
    mujoco, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    controller.reset(data)

    for _ in range(40):
        controller.apply_action(data, np.zeros(7))
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.ctrl))
    assert data.qpos[6:8] == pytest.approx([0.044, 0.044], abs=5e-3)


def test_zero_action_hold_does_not_sag_under_gravity():
    mujoco, model, data = _make_model_and_data()

    from mujoco_arx5.controllers import Arx5OperationalSpaceController

    controller = Arx5OperationalSpaceController(model, mode="train")
    controller.reset(data)
    initial_eef = controller.get_ee_pose(data).position.copy()

    for _ in range(120):
        controller.apply_target(data)
        mujoco.mj_step(model, data)

    final_eef = controller.get_ee_pose(data).position
    assert np.linalg.norm(final_eef - initial_eef) < 1.0e-5


def test_hdf5_action_replay_script_runs_without_viewer(tmp_path):
    dataset = tmp_path / "actions.hdf5"
    video_path = tmp_path / "replay.mp4"
    with h5py.File(dataset, "w") as h5_file:
        h5_file.attrs["source_env_origin"] = np.asarray([-1.5, 1.5, 0.0], dtype=np.float32)
        h5_file.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        obs_group = h5_file.create_group("obs")
        obs_group.create_dataset(
            "eef_pos",
            data=np.asarray([[-1.824, 1.2975, 1.1786], [-1.824, 1.2975, 1.1786]], dtype=np.float32),
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_arx5.replay_hdf5_actions",
            "--dataset",
            str(dataset),
            "--no-real-time",
            "--video-path",
            str(video_path),
            "--video-width",
            "160",
            "--video-height",
            "120",
            "--metrics-path",
            str(tmp_path / "metrics.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "arx5_robosuite_tabletop_dynamic.xml" in result.stdout
    assert "actions: 2 x 7" in result.stdout
    assert "cameras: external_camera, wrist_camera" in result.stdout
    assert "aligned robot root by" in result.stdout
    assert str(video_path) in result.stdout
    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert "eef position error vs hdf5" in result.stdout


def test_hdf5_action_replay_rejects_viewer_with_video(tmp_path):
    dataset = tmp_path / "actions.hdf5"
    with h5py.File(dataset, "w") as h5_file:
        h5_file.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_arx5.replay_hdf5_actions",
            "--dataset",
            str(dataset),
            "--viewer",
            "--video-path",
            str(tmp_path / "replay.mp4"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--viewer cannot be combined with video recording" in result.stderr

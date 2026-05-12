from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
RESET_STATES_CFG = (
    REPO_ROOT
    / "source"
    / "uwlab_tasks"
    / "uwlab_tasks"
    / "manager_based"
    / "manipulation"
    / "omnireset"
    / "config"
    / "arx5"
    / "reset_states_cfg.py"
)


def test_arx5_reset_state_collection_uses_current_table_and_workspace():
    cfg = RESET_STATES_CFG.read_text()

    assert "UWLAB_ASSETS_EXT_DIR" in cfg
    assert "props/robosuite_table/table.usd" in cfg
    assert "UWPatVention/pat_vention.usd" not in cfg
    assert "init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.799375)" in cfg
    assert "articulation_enabled=False" in cfg

    assert '"x": (-0.545, -0.525)' in cfg
    assert '"y": (-0.22, -0.20)' in cfg
    assert '"z": (0.79, 0.81)' in cfg

    assert '"x": (-0.4, -0.2)' in cfg
    assert '"y": (-0.3, -0.1)' in cfg
    assert '"z": (0.84, 0.84)' in cfg
    assert '"z": (0.84, 1.14)' in cfg


def test_arx5_reset_state_collection_preserves_omnireset_stage_semantics():
    cfg = RESET_STATES_CFG.read_text()

    assert "class ObjectAnywhereEEAnywhereEventCfg" in cfg
    assert "class ObjectRestingEEGraspedEventCfg" in cfg
    assert "class ObjectAnywhereEEGraspedEventCfg" in cfg
    assert "class ObjectPartiallyAssembledEEGraspedEventCfg" in cfg
    assert "reset_end_effector_pose_from_grasp_dataset" in cfg
    assert "reset_insertive_object_pose_from_partial_assembly_dataset" in cfg
    assert '"roll": (-np.pi, np.pi)' in cfg
    assert '"pitch": (-np.pi, np.pi)' in cfg
    assert '"yaw": (-np.pi, np.pi)' in cfg

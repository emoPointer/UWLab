from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts_cube_stack"
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
RL_STATE_CFG = (
    REPO_ROOT
    / "source"
    / "uwlab_tasks"
    / "uwlab_tasks"
    / "manager_based"
    / "manipulation"
    / "omnireset"
    / "config"
    / "arx5"
    / "rl_state_cfg.py"
)


def test_cube_stack_arx5_configs_use_robosuite_table_and_workspace():
    reset_cfg = RESET_STATES_CFG.read_text()
    rl_cfg = RL_STATE_CFG.read_text()

    for cfg in (reset_cfg, rl_cfg):
        assert "props/robosuite_table/table.usd" in cfg
        assert "UWPatVention/pat_vention.usd" not in cfg
        assert "init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.799375)" in cfg
        assert '"x": (-0.4, -0.2)' in reset_cfg
        assert '"y": (-0.3, -0.1)' in reset_cfg


def test_cube_stack_collection_scripts_pin_robosuite_workspace():
    common = (SCRIPTS_ROOT / "_common.sh").read_text()
    assert 'ROBOSUITE_WORKSPACE_X_MIN="${ROBOSUITE_WORKSPACE_X_MIN:-"-0.4"}"' in common
    assert 'ROBOSUITE_WORKSPACE_X_MAX="${ROBOSUITE_WORKSPACE_X_MAX:-"-0.2"}"' in common
    assert 'ROBOSUITE_WORKSPACE_Y_MIN="${ROBOSUITE_WORKSPACE_Y_MIN:-"-0.3"}"' in common
    assert 'ROBOSUITE_WORKSPACE_Y_MAX="${ROBOSUITE_WORKSPACE_Y_MAX:-"-0.1"}"' in common
    assert 'ROBOSUITE_TABLE_OBJECT_Z="${ROBOSUITE_TABLE_OBJECT_Z:-"0.84"}"' in common
    assert 'ROBOSUITE_OBJECT_AIR_Z_MAX="${ROBOSUITE_OBJECT_AIR_Z_MAX:-"1.14"}"' in common

    scripts = [
        SCRIPTS_ROOT / "03_reset_states_anywhere_ee_anywhere.sh",
        SCRIPTS_ROOT / "04_reset_states_anywhere_ee_grasped.sh",
        SCRIPTS_ROOT / "05_reset_states_resting_ee_grasped.sh",
        SCRIPTS_ROOT / "06_reset_states_partially_assembled_ee_grasped.sh",
    ]

    for script_path in scripts:
        script = script_path.read_text()
        assert "--task OmniReset-Arx5-" in script
        assert "env.scene.insertive_object=cube" in script
        assert "env.scene.receptive_object=cube" in script
        assert (
            'env.events.reset_receptive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]"'
            in script
        )
        assert (
            'env.events.reset_receptive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]"'
            in script
        )
        assert (
            'env.events.reset_receptive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_TABLE_OBJECT_Z]"'
            in script
        )

    for script_path in scripts[:2]:
        script = script_path.read_text()
        assert (
            'env.events.reset_insertive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]"'
            in script
        )
        assert (
            'env.events.reset_insertive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]"'
            in script
        )
        assert (
            'env.events.reset_insertive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_OBJECT_AIR_Z_MAX]"'
            in script
        )


def test_cube_stack_training_uses_local_recollected_cube_data():
    script = (SCRIPTS_ROOT / "07_train.sh").read_text()

    assert "--task OmniReset-Arx5-OSC-State-v0" in script
    assert "env.scene.insertive_object=cube" in script
    assert "env.scene.receptive_object=cube" in script
    assert "env.events.reset_from_reset_states.params.dataset_dir=\"$DATASET_DIR\"" in script
    assert "DATASET_DIR=\"${DATASET_DIR:-./Datasets/OmniReset}\"" in script

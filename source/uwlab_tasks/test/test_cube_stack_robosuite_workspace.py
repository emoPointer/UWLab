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

    grasp_script = (SCRIPTS_ROOT / "01_record_grasps.sh").read_text()
    assert 'DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"' in grasp_script
    assert 'NUM_GRASPS="${NUM_GRASPS:-1000}"' in grasp_script
    assert '--dataset_dir "$DATASET_DIR"' in grasp_script
    assert "--task OmniReset-Arx5-GraspSampling-v0" in grasp_script
    assert "env.scene.object=cube" in grasp_script

    partial_script = (SCRIPTS_ROOT / "02_record_partial_assemblies.sh").read_text()
    assert 'DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"' in partial_script
    assert 'NUM_PARTIAL_TRAJECTORIES="${NUM_PARTIAL_TRAJECTORIES:-10}"' in partial_script
    assert '--dataset_dir "$DATASET_DIR"' in partial_script
    assert "--task OmniReset-PartialAssemblies-v0" in partial_script
    assert "env.scene.insertive_object=cube" in partial_script
    assert "env.scene.receptive_object=cube" in partial_script

    scripts = [
        SCRIPTS_ROOT / "03_reset_states_anywhere_ee_anywhere.sh",
        SCRIPTS_ROOT / "04_reset_states_anywhere_ee_grasped.sh",
        SCRIPTS_ROOT / "05_reset_states_resting_ee_grasped.sh",
        SCRIPTS_ROOT / "06_reset_states_partially_assembled_ee_grasped.sh",
    ]

    for script_path in scripts:
        script = script_path.read_text()
        assert "--task OmniReset-Arx5-" in script
        assert 'DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"' in script
        assert 'NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"' in script
        assert '--dataset_dir "$DATASET_DIR"' in script
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

    collect_script = (SCRIPTS_ROOT / "03_collect_reset_states_for_vision_distill.sh").read_text()
    assert "03_reset_states_anywhere_ee_anywhere.sh" in collect_script
    assert "04_reset_states_anywhere_ee_grasped.sh" in collect_script
    assert "05_reset_states_resting_ee_grasped.sh" in collect_script
    assert "06_reset_states_partially_assembled_ee_grasped.sh" in collect_script
    assert 'DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES"' in collect_script

    teacher_data_script = (SCRIPTS_ROOT / "00_collect_state_teacher_data.sh").read_text()
    assert "01_record_grasps.sh" in teacher_data_script
    assert "02_record_partial_assemblies.sh" in teacher_data_script
    assert "03_collect_reset_states_for_vision_distill.sh" in teacher_data_script
    assert 'NUM_GRASPS="${NUM_GRASPS:-1000}"' in teacher_data_script
    assert 'NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"' in teacher_data_script

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
    init_py = (
        REPO_ROOT
        / "source"
        / "uwlab_tasks"
        / "uwlab_tasks"
        / "manager_based"
        / "manipulation"
        / "omnireset"
        / "config"
        / "arx5"
        / "__init__.py"
    ).read_text()
    rl_cfg = RL_STATE_CFG.read_text()

    assert "--task OmniReset-Arx5-OSC-CubeStack-State-v0" in script
    assert 'id="OmniReset-Arx5-OSC-CubeStack-State-v0"' in init_py
    assert "rl_state_cfg:Arx5OSCCubeStackTrainCfg" in init_py
    assert "class CubeStackTrainEventCfg(TrainEventCfg)" in rl_cfg
    assert "using reset states collected in the deploy/vision workspace" in rl_cfg
    assert "scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=128, env_spacing=3.0)" in rl_cfg
    assert "env.scene.insertive_object=cube" in script
    assert "env.scene.receptive_object=cube" in script
    assert "env.commands.task_command.success_mode=stack_center" in script
    assert "env.commands.task_command.success_orientation_required=false" in script
    assert 'SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-2.0}"' in script
    assert 'env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE"' in script
    assert "env.events.reset_from_reset_states.params.dataset_dir=\"$DATASET_DIR\"" in script
    assert "DATASET_DIR=\"${DATASET_DIR:-./Datasets/OmniReset}\"" in script

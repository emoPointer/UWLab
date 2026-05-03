from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
OMNIRESET_ROOT = (
    REPO_ROOT / "source" / "uwlab_tasks" / "uwlab_tasks" / "manager_based" / "manipulation" / "omnireset"
)


def test_deploy_fixed_receptive_pose_syncs_visual_root():
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "def _nearest_xformable_prim" in events
    assert "def _sync_rigid_root_visual_pose" in events
    assert "def _sync_articulation_root_visual_pose" in events
    assert "_sync_rigid_root_visual_pose(" in events
    assert "_nearest_xformable_prim(prim)" in events
    assert "standardize_xform_ops" in events


def test_rsl_rl_play_disable_fabric_flag_updates_env_cfg():
    play = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py").read_text()

    assert "env_cfg.sim.use_fabric = not args_cli.disable_fabric" in play


def test_arx5_deploy_scene_uses_robosuite_table():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()

    assert "UWLAB_ASSETS_EXT_DIR" in cfg
    assert "props/robosuite_table/table.usd" in cfg
    assert 'init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.799375)' in cfg
    assert "articulation_enabled=False" in cfg


def test_arx5_deploy_events_align_to_robosuite_lift_table_edge():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "align_deploy_scene_to_robosuite_table" in cfg
    assert "align_deploy_scene_to_robosuite_table" in events
    assert '"robosuite_robot_base_pose": (-0.535, -0.21, 0.8' in cfg
    assert '"training_robot_base_pose": (0.0, 0.0, 0.0' in cfg
    assert '"receptive_object_pose": (-0.30, -0.20, 0.84' in cfg


def test_arx5_deploy_randomizes_task_pair_in_robosuite_workspace():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()
    script = (REPO_ROOT / "scripts_peg_insertion" / "08_play_deploy_fixed.sh").read_text()

    assert '"workspace_x_range": (-0.4, -0.2)' in cfg
    assert '"workspace_y_range": (-0.3, -0.1)' in cfg
    assert "workspace_x_range" in events
    assert "workspace_y_range" in events
    assert "torch.empty(" in events
    assert "uniform_(float(x_min), float(x_max))" in events
    assert "uniform_(float(y_min), float(y_max))" in events
    assert "object_delta = receptive_root_pose[:, :3] - receptive_pose_after_base_align[:, :3]" in events
    assert "insertive_pose[:, :3] += object_delta" in events
    assert "log_every_reset" in events
    assert "[UWLab deploy reset]" in events
    assert "WORKSPACE_X_MIN" in script
    assert "WORKSPACE_Y_MAX" in script
    assert "params.workspace_x_range" in script
    assert "params.workspace_y_range" in script
    assert "DEPLOY_LOG_EVERY_RESET" in script
    assert "params.log_every_reset" in script
    assert 'SEED="${SEED:--1}"' in script
    assert '--seed "$SEED"' in script
    assert "extra_args=" in script

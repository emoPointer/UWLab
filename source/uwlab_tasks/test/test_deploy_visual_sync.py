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


def test_deploy_table_alignment_also_syncs_external_camera_anchor():
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()

    assert "def _sync_sibling_xform_pose" in events
    assert '"/World/envs/env_{env_id}/Table/external_cam"' in events
    assert "external_camera_table_relative_pose" in events
    assert "relative_pose=external_camera_table_relative_pose" in events
    assert (
        '"external_camera_table_relative_pose": (0.517, 0.327, 0.589, 0.3604, 0.2030, 0.5000, 0.7609)'
        in cfg
    )
    assert "type(current_translate)(*local_pos)" in events
    assert "type(current_orient)(*local_quat)" in events
    assert "type(current_orient)(local_quat)" not in events


def test_rsl_rl_play_disable_fabric_flag_updates_env_cfg():
    play = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py").read_text()

    assert "env_cfg.sim.use_fabric = not args_cli.disable_fabric" in play


def test_arx5_deploy_scene_uses_robosuite_table():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()

    assert "UWLAB_ASSETS_EXT_DIR" in cfg
    assert "props/robosuite_table/table.usd" in cfg
    assert "spawn=sim_utils.UsdFileCfg(" in cfg
    assert 'init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.799375)' in cfg


def test_arx5_deploy_scene_adds_robosuite_aligned_backdrop_curtains():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "curtain_back = RigidObjectCfg" in cfg
    assert "curtain_left = RigidObjectCfg" in cfg
    assert "curtain_right = RigidObjectCfg" in cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainBack"' in cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainLeft"' in cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainRight"' in cfg
    assert "collision_enabled=False" in cfg
    assert "size=(0.01, 1.6, 2.125)" in cfg
    assert cfg.count("size=(0.01, 2.1, 2.125)") == 2

    assert '"backdrop_asset_names": (' in cfg
    assert '"curtain_back"' in cfg
    assert '"curtain_left"' in cfg
    assert '"curtain_right"' in cfg
    assert '"backdrop_cfgs": (' not in cfg
    assert '"backdrop_table_relative_poses": (' in cfg
    assert "(-1.1, 0.0, -0.280375, 1.0, 0.0, 0.0, 0.0)" in cfg
    assert "(-0.05, 0.8, -0.280375, 0.707, 0.0, 0.0, -0.707)" in cfg
    assert "(-0.05, -0.8, -0.280375, 0.707, 0.0, 0.0, -0.707)" in cfg
    assert 'init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.1, 0.0, 0.519)' in cfg
    assert 'init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.8, 0.519)' in cfg
    assert 'init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, -0.8, 0.519)' in cfg

    assert "backdrop_asset_names" in events
    assert "backdrop_cfgs" not in events
    assert "backdrop_table_relative_poses" in events
    assert "table_root_pose[:, :3] + backdrop_relative_pose[:, :3]" in events


def test_arx5_deploy_scene_adds_robosuite_aligned_virtual_cameras():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()

    assert "external_camera = TiledCameraCfg(" in cfg
    assert "wrist_camera = TiledCameraCfg(" in cfg
    assert "external_cam_link = AssetBaseCfg" not in cfg
    assert 'prim_path="{ENV_REGEX_NS}/Table/external_cam/Camera"' in cfg
    assert 'prim_path="{ENV_REGEX_NS}/Robot/camera/Camera"' in cfg
    assert "offset=TiledCameraCfg.OffsetCfg" not in cfg
    assert 'convention="ros"' not in cfg
    assert 'convention="opengl"' not in cfg
    assert "height=ROBOSUITE_CAMERA_HEIGHT" in cfg
    assert "width=ROBOSUITE_CAMERA_WIDTH" in cfg
    assert cfg.count("spawn=None") >= 2


def test_mjcf_converter_preserves_camera_only_links_as_xforms():
    converter = (REPO_ROOT / "scripts" / "tools" / "convert_mjcf.py").read_text()

    assert "def _add_camera_only_links" in converter
    assert "UsdGeom.Xform.Define(stage, external_cam_path)" in converter
    assert "xform.GetPrim().SetActive(True)" in converter
    assert "duplicate_body_path" in converter
    assert 'body.get("name")' in converter
    assert '"cam" in body_name.lower()' in converter
    assert "UsdPhysics.RigidBodyAPI" not in converter


def test_robosuite_table_usd_has_external_cam_anchor_at_mjcf_pose():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(REPO_ROOT / "source/uwlab_assets/uwlab_assets/props/robosuite_table/table.usd"))
    external_cam = stage.GetPrimAtPath("/table_only_with_external_cam/external_cam")

    assert external_cam.IsValid()
    assert external_cam.IsActive()
    assert external_cam.GetTypeName() == "Xform"
    assert not external_cam.HasAPI(UsdPhysics.RigidBodyAPI)

    duplicate_external_cam_body = stage.GetPrimAtPath("/table_only_with_external_cam/external_cam/external_cam")
    assert duplicate_external_cam_body.IsValid()
    assert not duplicate_external_cam_body.IsActive()
    assert not stage.GetPrimAtPath("/table_only_with_external_cam/worldBody/external_cam").IsValid()

    transform = UsdGeom.Xformable(external_cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert tuple(round(transform[3][i], 4) for i in range(3)) == (0.517, 0.327, 1.364)

    rigid_paths = [prim.GetPath().pathString for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert rigid_paths == ["/table_only_with_external_cam/table/table"]


def test_arx5_deploy_randomizes_backdrop_pose_and_color_each_reset():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()
    assert '"backdrop_position_jitter_m": 0.02' in cfg
    assert '"backdrop_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))' in cfg

    assert "backdrop_position_jitter_m" in events
    assert "backdrop_group_jitter = torch.empty(" in events
    assert "-float(backdrop_position_jitter_m)" in events
    assert "float(backdrop_position_jitter_m)" in events
    assert "backdrop_pose[:, :3] += backdrop_group_jitter" in events
    assert "torch.empty_like(backdrop_pose[:, :3]).uniform_(" not in events
    assert "backdrop_color_range" in events
    assert "shared_backdrop_colors = None" in events
    assert "shared_backdrop_colors = color_low + torch.rand(" in events
    assert "torch.rand((len(env_ids), 3)" in events
    assert "_set_rigid_root_visual_color(" in events
    assert "Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))" in events


def test_arx5_deploy_randomizes_peg_and_peghole_color_each_reset():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert '"task_object_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))' in cfg
    assert "task_object_color_range" in events
    assert "shared_task_object_colors = None" in events
    assert "shared_task_object_colors = color_low + torch.rand(" in events
    assert "_set_rigid_root_visual_color(insertive_object, shared_task_object_colors, env_ids)" in events
    assert "_set_rigid_root_visual_color(receptive_object, shared_task_object_colors, env_ids)" in events


def test_arx5_deploy_events_align_to_robosuite_lift_table_edge():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "align_deploy_scene_to_robosuite_table" in cfg
    assert "align_deploy_scene_to_robosuite_table" in events
    assert '"robosuite_robot_base_pose": (-0.535, -0.21, 0.8' in cfg
    assert '"training_robot_base_pose": (-0.535, -0.21, 0.8' in cfg
    assert '"receptive_object_pose": (-0.30, -0.20, 0.84' in cfg


def test_arx5_deploy_randomizes_robot_base_xy_only():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert '"robot_xy_jitter_m": 0.02' in cfg
    assert "camera_position_jitter_m" not in cfg
    assert "camera_orientation_jitter_deg" not in cfg
    assert "camera_path_templates" not in cfg

    assert "robot_xy_jitter_m: float = 0.0" in events
    assert "robot_xy_jitter[:, :2]" in events
    assert "robot_pose[:, :2] += robot_xy_jitter[:, :2]" in events
    assert "robot_root_pose[:, :2] += robot_xy_jitter[:, :2]" in events
    assert "robot_pose[:, 2]" not in events
    assert "robot_root_pose[:, 2]" not in events
    assert "def _jitter_authored_camera_poses" not in events


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
    assert 'WORKSPACE_X_MIN="${WORKSPACE_X_MIN:-"-0.4"}"' in script
    assert 'WORKSPACE_X_MAX="${WORKSPACE_X_MAX:-"-0.2"}"' in script
    assert 'WORKSPACE_Y_MIN="${WORKSPACE_Y_MIN:-"-0.3"}"' in script
    assert 'WORKSPACE_Y_MAX="${WORKSPACE_Y_MAX:-"-0.1"}"' in script
    assert "params.training_robot_base_pose" in script
    assert "params.workspace_x_range" in script
    assert "params.workspace_y_range" in script
    assert "DEPLOY_LOG_EVERY_RESET" in script
    assert "params.log_every_reset" in script
    assert 'SEED="${SEED:--1}"' in script
    assert '--seed "$SEED"' in script
    assert "extra_args=" in script


def test_deploy_play_script_supports_recording_virtual_cameras_until_first_reset():
    script = (REPO_ROOT / "scripts_peg_insertion" / "08_play_deploy_fixed.sh").read_text()

    assert 'RECORD_DEPLOY_CAMERAS_UNTIL_RESET="${RECORD_DEPLOY_CAMERAS_UNTIL_RESET:-false}"' in script
    assert 'DEPLOY_CAMERA_OUTPUT_DIR="${DEPLOY_CAMERA_OUTPUT_DIR:-}"' in script
    assert "--record_deploy_cameras_until_reset" in script
    assert "--deploy_camera_output_dir" in script
    assert 'camera_args=()' in script


def test_arx5_deploy_rejects_initial_successful_resets():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "reject_initial_successful_resets = EventTerm" in cfg
    assert "func=task_mdp.RejectInitialAssemblySuccessReset" in cfg
    assert '"reset_event_name": "reset_from_reset_states"' in cfg
    assert '"align_event_name": "align_deploy_scene_to_robosuite_table"' in cfg
    assert '"max_resample_attempts": 20' in cfg

    assert "class RejectInitialAssemblySuccessReset" in events
    assert "success_position_threshold" in events
    assert "success_orientation_threshold" in events
    assert "env.event_manager.get_term_cfg(self.reset_event_name)" in events
    assert "reset_params[\"success\"] = None" in events
    assert "initial_success_mask" in events


def test_arx5_deploy_ends_rollout_after_stable_success():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()

    assert "class DeployTerminationsCfg(TerminationsCfg)" in cfg
    assert "success = DoneTerm(" in cfg
    assert "func=task_mdp.consecutive_success_state_with_min_length" in cfg
    assert '"num_consecutive_successes": 10' in cfg
    assert '"min_episode_length": 10' in cfg
    assert "terminations: DeployTerminationsCfg = DeployTerminationsCfg()" in cfg


def test_arx5_finetune_uses_fixed_robot_workspace_task_pair_reset():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert "class FixedRobotWorkspaceTaskPairReset" in events
    assert "class FinetuneEventCfg(BaseEventCfg)" in cfg
    assert "func=task_mdp.FixedRobotWorkspaceTaskPairReset" in cfg
    assert "class FinetuneEventCfg(TrainEventCfg)" not in cfg
    assert '"robot_pose": (-0.535, -0.21, 0.8' in cfg
    assert '"table_pose": (0.0, 0.0, 0.799375' in cfg
    assert '"insertive_object_pose": (-0.30, -0.20, 0.87' in cfg
    assert '"workspace_x_range": (-0.4, -0.2)' in cfg
    assert '"workspace_y_range": (-0.3, -0.1)' in cfg
    assert '"insertive_workspace_x_range": (-0.4, -0.2)' in cfg
    assert '"insertive_workspace_y_range": (-0.3, -0.1)' in cfg
    assert '"success": "env.reward_manager.get_term_cfg(\'progress_context\').func.success"' in cfg
    assert "self.success_monitor.success_update" in events
    assert 'f"Metrics/task_0_success_rate"' in events


def test_arx5_finetune_reset_disables_usd_visual_sync_for_headless_training():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    events = (OMNIRESET_ROOT / "mdp" / "events.py").read_text()

    assert '"sync_visuals": False' in cfg
    assert "sync_visuals: bool = True" in events
    assert "if sync_visuals:" in events


def test_arx5_finetune_pads_table_material_properties_for_old_critic_checkpoint():
    cfg = (OMNIRESET_ROOT / "config" / "arx5" / "rl_state_cfg.py").read_text()
    observations = (OMNIRESET_ROOT / "mdp" / "observations.py").read_text()

    assert "get_material_properties_compat" in cfg
    assert '"output_dim": 21' in cfg
    assert ".repeat(" in observations
    assert "new_zeros" not in observations


def test_rsl_rl_train_does_not_expose_actor_freeze_warmup():
    train = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py").read_text()

    assert "freeze_actor" not in train
    assert "actor_freeze" not in train

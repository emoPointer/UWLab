from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "scripts_cube_stack" / "08_collect_state_policy_dataset.py"
WRAPPER = REPO_ROOT / "scripts_cube_stack" / "08_collect_state_policy_dataset.sh"
REPLAY_SCRIPT = REPO_ROOT / "scripts_cube_stack" / "09_replay_state_policy_dataset.py"
REPLAY_WRAPPER = REPO_ROOT / "scripts_cube_stack" / "09_replay_state_policy_dataset.sh"


def test_cube_stack_policy_dataset_collection_script_is_independent_and_camera_enabled():
    script = SCRIPT.read_text()

    assert "scripts/reinforcement_learning/rsl_rl/play.py" not in script
    assert "AppLauncher.add_app_launcher_args(parser)" in script
    assert "args_cli.enable_cameras = True" in script
    assert 'default="OmniReset-Arx5-OSC-State-Deploy-Play-v0"' in script
    assert 'parser.add_argument("--env_spacing", type=float, default=3.0)' in script
    assert "env_cfg.scene.env_spacing = args_cli.env_spacing" in script
    assert "DEFAULT_HYDRA_OVERRIDES" in script
    assert '"env.scene.insertive_object=cube"' in script
    assert '"env.scene.receptive_object=cube"' in script
    assert "def _set_fixed_robot_workspace_reset" in script
    assert "func=task_mdp.FixedRobotWorkspaceTaskPairReset" in script
    assert '"robot_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0)' in script
    assert '"robot_xy_jitter_m": 0.0' in script
    assert "camera_position_jitter_m" not in script
    assert "camera_orientation_jitter_deg" not in script
    assert "camera_path_templates" not in script
    assert '"workspace_x_range": (-0.4, -0.2)' in script
    assert '"workspace_y_range": (-0.3, -0.1)' in script
    assert "_set_fixed_robot_workspace_reset(env_cfg)" in script


def test_cube_stack_policy_dataset_uses_fixed_semantic_cube_colors():
    collect_script = SCRIPT.read_text()
    replay_script = REPLAY_SCRIPT.read_text()
    events = (
        REPO_ROOT
        / "source"
        / "uwlab_tasks"
        / "uwlab_tasks"
        / "manager_based"
        / "manipulation"
        / "omnireset"
        / "mdp"
        / "events.py"
    ).read_text()

    assert "insertive_object_color: tuple[float, float, float] | None = None" in events
    assert "receptive_object_color: tuple[float, float, float] | None = None" in events
    assert "_set_rigid_root_visual_color(insertive_object, insertive_colors, env_ids)" in events
    assert "_set_rigid_root_visual_color(receptive_object, receptive_colors, env_ids)" in events

    for script in (collect_script, replay_script):
        assert 'align_params["task_object_color_range"] = None' in script
        assert 'align_params["insertive_object_color"] = (0.0, 1.0, 0.0)' in script
        assert 'align_params["receptive_object_color"] = (1.0, 0.0, 0.0)' in script


def test_cube_stack_policy_dataset_collection_hdf5_schema():
    script = SCRIPT.read_text()

    assert "import h5py" in script
    assert "import imageio.v2 as imageio" in script
    assert "def _demo_output_path" in script
    assert "def _demo_video_path" in script
    assert "def _write_demo_video" in script
    assert '"data/demo_0"' not in script
    assert 'obs_group = h5_file.create_group("obs")' in script
    assert 'h5_file.create_dataset("actions"' in script
    assert '"actions"' in script
    assert '"table_cam"' in script
    assert '"wrist_cam"' in script
    assert '"eef_pos"' in script
    assert '"eef_rot_6d"' in script
    assert "eef_quat" not in script
    assert "matrix_from_quat" in script
    assert ".transpose(1, 2).reshape" in script
    assert '"insertive_cube_pos"' in script
    assert '"insertive_cube_quat"' in script
    assert '"receptive_cube_pos"' in script
    assert '"receptive_cube_quat"' in script
    assert 'h5_file.attrs["control_frequency_hz"]' in script
    assert 'h5_file.attrs["demo_index"]' in script
    assert 'h5_file.attrs["source_env_id"]' in script
    assert 'h5_file.attrs["source_env_origin"]' in script
    assert 'h5_file.attrs["env_spacing"]' in script
    assert 'h5_file.attrs["num_demos"] = 1' in script
    assert 'h5_file.attrs["success"]' in script
    assert 'h5_file.attrs["table_cam_video"]' in script
    assert "imageio.get_writer" in script
    assert "episode.table_cam" in script
    assert "fps=control_frequency_hz" in script


def test_cube_stack_policy_dataset_collection_records_policy_and_scene_state_per_env():
    script = SCRIPT.read_text()

    assert "actions = policy(obs)" in script
    assert '_find_single_body_id(unwrapped_env.scene["robot"], args_cli.ee_body_name)' in script
    assert "camera_name_to_hdf5_key" in script
    assert '"external_camera": "table_cam"' in script
    assert '"wrist_camera": "wrist_cam"' in script
    assert 'scene["insertive_object"].data.root_pos_w' in script
    assert 'scene["receptive_object"].data.root_pos_w' in script
    assert "for env_id in done_env_ids:" in script
    assert "episode_buffers[env_id].reset()" in script
    assert "if success or args_cli.save_failed:" in script
    assert "demo_output_path = _demo_output_path(output_path, saved_demos)" in script
    assert "demo_video_path = _demo_video_path(demo_output_path)" in script
    assert "_write_demo_video(demo_video_path, episode_buffers[env_id].table_cam, fps=control_frequency_hz)" in script
    assert "env_origins_np = _to_numpy(unwrapped_env.scene.env_origins)" in script
    assert "source_env_origin=env_origins_np[env_id]" in script
    assert 'f"[INFO] saved {demo_output_path}: env={env_id} "' in script


def test_cube_stack_policy_dataset_collection_wrapper_defaults():
    wrapper = WRAPPER.read_text()

    assert 'source "$(dirname "$0")/_common.sh"' in wrapper
    assert 'NUM_ENVS="${NUM_ENVS:-4}"' in wrapper
    assert 'ENV_SPACING="${ENV_SPACING:-3.0}"' in wrapper
    assert 'NUM_DEMOS="${NUM_DEMOS:-50}"' in wrapper
    assert 'OUTPUT_FILE="${OUTPUT_FILE:-./datasets/cube_stack_state_policy.hdf5}"' in wrapper
    assert "08_collect_state_policy_dataset.py" in wrapper
    assert '--env_spacing "$ENV_SPACING"' in wrapper
    assert "--headless" in wrapper


def test_cube_stack_policy_dataset_replay_script_uses_recorded_actions_and_object_poses():
    script = REPLAY_SCRIPT.read_text()

    assert "Collect cube-stack state-policy rollouts" not in script
    assert "Replay collected cube-stack state-policy HDF5 rollouts" in script
    assert 'default="/home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_*.hdf5"' in script
    assert '"actions"' in script
    assert '"obs/insertive_cube_pos"' in script
    assert '"obs/insertive_cube_quat"' in script
    assert '"obs/receptive_cube_pos"' in script
    assert '"obs/receptive_cube_quat"' in script
    assert "def _reset_to_recorded_initial_state" in script
    assert "def _infer_source_env_origin" in script
    assert 'if "source_env_origin" in h5_file.attrs:' in script
    assert "target_pos = pos - source_env_origin + unwrapped_env.scene.env_origins[0].detach().cpu().numpy()" in script
    assert "if not (-0.45 <= local_x <= -0.15 and -0.35 <= local_y <= -0.05):" in script
    assert 'unwrapped_env.scene["insertive_object"]' not in script
    assert '_write_object_pose(' in script
    assert 'env.step(action.unsqueeze(0))' in script
    assert 'if args_cli.num_envs != 1:' in script
    assert "checkpoint" not in script
    assert '"robot_xy_jitter_m": 0.0' in script
    assert 'align_params["robot_xy_jitter_m"] = 0.0' in script
    assert "import imageio.v2 as imageio" in script
    assert 'parser.add_argument("--video_path", type=str, default="./videos/cube_stack_replays")' in script
    assert 'parser.add_argument("--camera_name", type=str, default="external_camera")' in script
    assert 'parser.add_argument("--metrics_path", type=str, default=None)' in script
    assert "def _camera_rgb" in script
    assert "def _write_video" in script
    assert "def _summarize_eef_error" in script
    assert "imageio.get_writer" in script
    assert '"obs/eef_pos"' in script
    assert "def _resolve_dataset_files" in script
    assert "glob.glob" in script
    assert "def _video_output_path" in script
    assert "for file_index, dataset_file in enumerate(dataset_files):" in script


def test_cube_stack_policy_dataset_replay_wrapper_defaults():
    wrapper = REPLAY_WRAPPER.read_text()

    assert 'source "$(dirname "$0")/_common.sh"' in wrapper
    assert "cube_stack_state_policy_demo_*.hdf5" in wrapper
    assert "09_replay_state_policy_dataset.py" in wrapper
    assert "--dataset_file" in wrapper
    assert "--num_envs 1" in wrapper
    assert 'VIDEO_PATH="${VIDEO_PATH:-./videos/cube_stack_replays}"' in wrapper
    assert '--video_path "$VIDEO_PATH"' in wrapper

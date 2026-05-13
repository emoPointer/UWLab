from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
ARX5_CONFIG_DIR = (
    REPO_ROOT
    / "source"
    / "uwlab_tasks"
    / "uwlab_tasks"
    / "manager_based"
    / "manipulation"
    / "omnireset"
    / "config"
    / "arx5"
)
UWLAB_RL_DIR = REPO_ROOT / "source" / "uwlab_rl" / "uwlab_rl" / "rsl_rl"


def test_arx5_vision_task_is_registered_with_vision_distill_agent():
    init_py = (ARX5_CONFIG_DIR / "__init__.py").read_text()

    assert 'id="OmniReset-Arx5-OSC-Vision-v0"' in init_py
    assert 'id="OmniReset-Arx5-OSC-Vision-Play-v0"' in init_py
    assert "rl_vision_cfg:Arx5OSCVisionTrainCfg" in init_py
    assert "rl_vision_cfg:Arx5OSCVisionPlayCfg" in init_py
    assert "rsl_rl_vision_cfg:VisionDistill_PPORunnerCfg" in init_py


def test_arx5_vision_observations_match_v1_plan():
    rl_vision_cfg = (ARX5_CONFIG_DIR / "rl_vision_cfg.py").read_text()
    rl_state_cfg = (ARX5_CONFIG_DIR / "rl_state_cfg.py").read_text()

    assert "class VisionSceneCfg(DeployRlStateSceneCfg)" in rl_vision_cfg
    assert 'prim_path="{ENV_REGEX_NS}/Table/external_cam/Camera"' in rl_state_cfg
    assert 'prim_path="{ENV_REGEX_NS}/Robot/camera/Camera"' in rl_state_cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainBack"' in rl_state_cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainLeft"' in rl_state_cfg
    assert 'prim_path="{ENV_REGEX_NS}/CurtainRight"' in rl_state_cfg
    assert "class VisionObservationsCfg" in rl_vision_cfg
    assert "joint_pos = ObsTerm(func=task_mdp.joint_pos)" in rl_vision_cfg
    assert "external_rgb = ObsTerm" in rl_vision_cfg
    assert '"crop_size": 400' in rl_vision_cfg
    assert '"output_size": (128, 128)' in rl_vision_cfg
    assert "wrist_rgb = ObsTerm" in rl_vision_cfg
    assert '"crop_size": None' in rl_vision_cfg
    assert "self.concatenate_terms = False" in rl_vision_cfg
    assert "self.history_length = 1" in rl_vision_cfg
    assert "self.flatten_history_dim = False" in rl_vision_cfg
    assert "class TeacherPolicyCfg(ObservationsCfg.PolicyCfg)" in rl_vision_cfg
    assert "teacher_policy: TeacherPolicyCfg = TeacherPolicyCfg()" in rl_vision_cfg


def test_arx5_vision_training_randomizes_backdrop_and_light_on_reset():
    rl_vision_cfg = (ARX5_CONFIG_DIR / "rl_vision_cfg.py").read_text()
    events_py = (
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

    assert "class VisionTrainEventCfg(TrainEventCfg)" in rl_vision_cfg
    assert "events: VisionTrainEventCfg = VisionTrainEventCfg()" in rl_vision_cfg
    assert "randomize_backdrop_visuals = EventTerm" in rl_vision_cfg
    assert "func=task_mdp.randomize_backdrop_visuals" in rl_vision_cfg
    assert '"backdrop_asset_names": (' in rl_vision_cfg
    assert '"curtain_back"' in rl_vision_cfg
    assert '"curtain_left"' in rl_vision_cfg
    assert '"curtain_right"' in rl_vision_cfg
    assert '"backdrop_position_jitter_m": 0.02' in rl_vision_cfg
    assert '"backdrop_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))' in rl_vision_cfg
    assert "randomize_sky_light = EventTerm" in rl_vision_cfg
    assert "func=task_mdp.randomize_dome_light" in rl_vision_cfg
    assert '"intensity_range": (800.0, 3500.0)' in rl_vision_cfg
    assert '"rotation_range": (0.0, 360.0)' in rl_vision_cfg
    assert '"pitch_range": (-10.0, 10.0)' in rl_vision_cfg
    assert '"roll_range": (-5.0, 5.0)' in rl_vision_cfg
    assert "def randomize_backdrop_visuals(" in events_py
    assert "def randomize_dome_light(" in events_py
    assert "_apply_dome_light_rotation(light_prim, rotation_range)" in events_py


def test_vision_distillation_agent_cfg_uses_resnet18_and_online_teacher_loss():
    agent_cfg = (ARX5_CONFIG_DIR / "agents" / "rsl_rl_vision_cfg.py").read_text()

    assert 'class_name = "VisionDistillOnPolicyRunner"' in agent_cfg
    assert 'experiment_name = "arx5_omnireset_vision_distill"' in agent_cfg
    assert 'wandb_project = "arx5_vision_distill"' in agent_cfg
    assert "wandb_camera_video_interval = 100" in agent_cfg
    assert 'wandb_camera_video_camera_names = ["external_camera"]' in agent_cfg
    assert 'obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in agent_cfg
    assert 'name="resnet18"' in agent_cfg
    assert "pretrained=False" in agent_cfg
    assert "share_camera_encoder=False" in agent_cfg
    assert "feature_dim=128" in agent_cfg
    assert "RslRlVisionDistillPpoAlgorithmCfg" in agent_cfg
    assert 'teacher_checkpoint=""' in agent_cfg
    assert 'teacher_obs_group="teacher_policy"' in agent_cfg
    assert 'loss_type="mse"' in agent_cfg
    assert "lambda_initial=1.0" in agent_cfg
    assert "lambda_final=0.05" in agent_cfg
    assert "decay_iterations=8000" in agent_cfg


def test_vision_distillation_runner_and_ppo_are_integrated_with_train_play():
    train_py = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py").read_text()
    play_py = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py").read_text()
    runner = (UWLAB_RL_DIR / "vision_distill_runner.py").read_text()
    ppo = (UWLAB_RL_DIR / "vision_distill_ppo.py").read_text()
    actor_critic = (UWLAB_RL_DIR / "vision_actor_critic.py").read_text()

    assert "VisionDistillOnPolicyRunner" in train_py
    assert "VisionDistillOnPolicyRunner" in play_py
    assert "Skipping flat JIT/ONNX export for structured vision policy" in play_py
    assert "class VisionDistillOnPolicyRunner(OnPolicyRunner)" in runner
    assert "ActorCritic(" in runner
    assert "teacher_checkpoint" in runner
    assert "teacher.load_state_dict" in runner
    assert "No teacher checkpoint configured; using inference-only placeholder teacher" in runner
    assert "def _log_wandb_camera_videos" in runner
    assert "wandb.Video(video, fps=fps, format=\"mp4\")" in runner
    assert 'f"train_camera/{camera_name}"' in runner
    assert "class VisionDistillPPO(PPO)" in ppo
    assert "class NestedObservationRolloutStorage(RolloutStorage)" in ppo
    assert "def init_storage(" in ppo
    assert "NestedObservationRolloutStorage(" in ppo
    assert "obs.unsqueeze(0).expand" in ppo
    assert "teacher.act_inference(obs_batch)" in ppo
    assert "torch.nn.functional.mse_loss(mu_batch, teacher_actions)" in ppo
    assert "teacher_student_action_l2" in ppo
    assert "class VisionActorCritic" in actor_critic
    assert "resnet18" in actor_critic
    assert "supports_flat_export: bool = False" in actor_critic


def test_cube_stack_vision_distillation_training_script_exposes_hydra_overrides():
    script = (REPO_ROOT / "scripts_cube_stack" / "07_train_vision_distill.sh").read_text()

    assert "TEACHER_CKPT" in script
    assert "--task OmniReset-Arx5-OSC-Vision-v0" in script
    assert "--enable_cameras" in script
    assert 'LOGGER="${LOGGER:-wandb}"' in script
    assert 'WANDB_CAMERA_VIDEO_INTERVAL="${WANDB_CAMERA_VIDEO_INTERVAL:-100}"' in script
    assert 'agent.wandb_camera_video_interval="$WANDB_CAMERA_VIDEO_INTERVAL"' in script
    assert "agent.wandb_camera_video_camera_names='[external_camera]'" in script
    assert 'agent.algorithm.teacher_checkpoint="$TEACHER_CKPT"' in script
    assert 'agent.algorithm.distillation.lambda_initial="$DISTILL_LAMBDA_INITIAL"' in script
    assert 'agent.algorithm.distillation.lambda_final="$DISTILL_LAMBDA_FINAL"' in script
    assert "env.scene.insertive_object=cube" in script
    assert "env.scene.receptive_object=cube" in script
    assert "insertive_object_color='[0.0,1.0,0.0]'" in script
    assert "receptive_object_color='[1.0,0.0,0.0]'" in script

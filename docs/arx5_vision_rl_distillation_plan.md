# ARX5 Vision RL Distillation Plan

This plan defines the first version of distilling the existing ARX5 state-based RL policy into a perception-based RL policy. The goal is not offline imitation learning. The student is trained online with PPO in Isaac/PhysX, while a frozen state-policy teacher provides action regularization.

## Goal

Train a deployable perception policy with:

- `policy obs`: `joint_pos`, `external_rgb`, `wrist_rgb`
- `action`: single-step 7D action, same as current ARX5 OSC policy
- `critic obs`: current privileged critic observations
- `teacher`: frozen existing state-based policy checkpoint
- `algorithm`: online PPO + teacher action distillation loss

The first version deliberately does not use action chunks. Chunked actions change the environment action abstraction and should be a second-stage feature after the single-action vision baseline is stable.

## V1 Task Scope

The first version targets the cube stack task:

```text
env.scene.insertive_object=cube
env.scene.receptive_object=cube
```

The workspace must stay aligned with the existing cube-stack reset scripts and Robosuite-style table setup:

```text
robot_pose = (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0)
table_pose = (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0)
workspace_x_range = (-0.4, -0.2)
workspace_y_range = (-0.3, -0.1)
```

Cube colors are a prerequisite for vision training. The two cubes must have fixed semantic colors:

```text
insertive/top cube: green (0.0, 1.0, 0.0)
receptive/bottom cube: red (1.0, 0.0, 0.0)
```

This should be enforced in the Isaac reset path, not only in replay scripts, so training cameras always see the same task semantics. The same semantic colors should also be used by MuJoCo render-only replay to keep videos and diagnostics consistent.

## Current UWLab Baseline

The existing ARX5 state policy is configured in:

- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_state_cfg.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/agents/rsl_rl_cfg.py`

The current state-policy actor observes privileged object state, including:

- `prev_actions`
- `joint_pos`
- `end_effector_pose`
- `insertive_asset_pose`
- `receptive_asset_pose`
- `insertive_asset_in_receptive_asset_frame`

The current critic already uses privileged observations and should be reused as the student critic observation definition.

The current action is:

- `arm`: 6D relative OSC pose action
- `gripper`: 1D binary gripper action
- total action dimension: 7

## First-Version Design

### Student Actor Observations

The student actor gets exactly these three terms:

```text
joint_pos
external_rgb
wrist_rgb
```

No object pose, no end-effector pose, no previous action in the policy observation for v1.

### Image Preprocessing

The camera inputs are:

```text
wrist_rgb:
  source: wrist_camera rgb
  transform: resize full frame to 128x128
  output: float32 NCHW, normalized to [0, 1]

external_rgb:
  source: external_camera rgb
  source frame: expected 640x480
  crop: top-right 400x400 pixels
  pixel range: y = [0, 400), x = [width - 400, width)
  for 640x480: y = [0, 400), x = [240, 640)
  transform: resize crop to 128x128
  output: float32 NCHW, normalized to [0, 1]
```

The crop operation should be implemented as a proper observation function, not as an ad hoc transform inside the training loop. Suggested addition:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/observations.py
  process_image_crop_resize(...)
```

Hydra-configurable parameters:

```text
env.observations.policy.external_rgb.params.crop_top=0
env.observations.policy.external_rgb.params.crop_right=0
env.observations.policy.external_rgb.params.crop_size=400
env.observations.policy.external_rgb.params.output_size=[128, 128]
env.observations.policy.wrist_rgb.params.output_size=[128, 128]
```

### Vision Encoder

Use ResNet18 as the vision encoder.

Recommended v1 architecture:

```text
external_rgb -> ResNet18 encoder -> external_feature
wrist_rgb    -> ResNet18 encoder -> wrist_feature
joint_pos    -> proprio MLP      -> proprio_feature

concat(external_feature, wrist_feature, proprio_feature)
  -> actor MLP
  -> Gaussian action distribution
  -> 7D action
```

Implementation choice:

- Use separate ResNet18 encoders for external and wrist camera in v1.
- Start with `pretrained=False` for reproducibility and no network dependency.
- Keep `pretrained=True` or local weights as a Hydra option for later.
- Replace the ResNet classifier head with `Identity`.
- Add a small projection layer after each ResNet feature, for example `512 -> 128`.

Hydra-configurable parameters:

```text
agent.policy.class_name=VisionActorCritic
agent.policy.vision_encoder.name=resnet18
agent.policy.vision_encoder.pretrained=false
agent.policy.vision_encoder.share_camera_encoder=false
agent.policy.vision_encoder.feature_dim=128
agent.policy.proprio_mlp.hidden_dims=[128,128]
agent.policy.proprio_mlp.feature_dim=64
agent.policy.actor_hidden_dims=[512,256,128]
agent.policy.critic_hidden_dims=[512,256,128,64]
```

### Student Critic

The student critic keeps the current privileged critic observation structure:

```text
privileged critic obs -> MLP -> value
```

The critic is not frozen. It should be trainable during student PPO.

Recommended checkpoint behavior:

- Warm-start student critic from the state-policy checkpoint if and only if critic observation dimensions are unchanged.
- Continue training the critic after warm-start.
- Load critic normalizer from the teacher checkpoint when compatible.
- Do not freeze old critic weights.

Reason: the student actor will induce a different state distribution from the teacher actor, especially early in training. A frozen teacher critic can give biased advantages.

### Teacher Policy

The teacher is the existing state-based policy checkpoint.

Teacher properties:

- frozen
- no gradient
- receives state-policy actor observations
- outputs 7D action
- only used during training
- not exported for deployment

The environment needs an additional observation group for the teacher, separate from student `policy`:

```text
teacher_policy:
  prev_actions
  joint_pos
  end_effector_pose
  insertive_asset_pose
  receptive_asset_pose
  insertive_asset_in_receptive_asset_frame
```

This group should match the old state-policy actor observation exactly, including history behavior and normalization assumptions, so teacher inference matches the checkpoint.

### Distillation Loss

Use online teacher action regularization:

```text
loss = PPO_loss(student) + lambda_distill * action_distill_loss
```

For v1:

```text
action_distill_loss = mean_squared_error(student_mean_action, teacher_action)
```

Use actor mean rather than sampled action for the student target loss, so the regularizer is low variance.

Distillation coefficient schedule:

```text
lambda_distill initial: 1.0
lambda_distill final: 0.05 or 0.0
decay: linear over 20-40% of total iterations
```

Hydra-configurable parameters:

```text
agent.algorithm.teacher_checkpoint=/path/to/model_xxxx.pt
agent.algorithm.distillation.enabled=true
agent.algorithm.distillation.loss_type=mse
agent.algorithm.distillation.lambda_initial=1.0
agent.algorithm.distillation.lambda_final=0.05
agent.algorithm.distillation.decay_iterations=8000
agent.algorithm.distillation.teacher_obs_group=teacher_policy
```

## Required Code Additions

### 1. Vision Observation Config

Add:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_vision_cfg.py
```

Contents:

- `Arx5VisionSceneCfg`
- `VisionObservationsCfg`
- `VisionObservationsCfg.PolicyCfg`
- `VisionObservationsCfg.CriticCfg`
- `VisionObservationsCfg.TeacherPolicyCfg`
- `Arx5OSCVisionTrainCfg`
- `Arx5OSCVisionPlayCfg`

`PolicyCfg` should set:

```text
concatenate_terms = False
history_length = 1
enable_corruption = True
```

`CriticCfg` should remain concatenated:

```text
concatenate_terms = True
history_length = 1
enable_corruption = False
```

`TeacherPolicyCfg` should match the old state actor observation.

### 2. Image Crop Observation Function

Add to:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/observations.py
```

Function:

```text
process_image_crop_resize(
    env,
    sensor_cfg,
    data_type="rgb",
    crop_top=0,
    crop_left=None,
    crop_right=0,
    crop_size=400,
    output_size=(128, 128),
    normalize=True,
)
```

Behavior:

- read `sensor.data.output["rgb"]`
- crop before resize
- if `crop_left is None`, compute `crop_left = width - crop_right - crop_size`
- return `float32` tensor in `NCHW`
- values in `[0, 1]`
- validate crop is inside image bounds

### 3. Task Registration

Modify:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/__init__.py
```

Add task ids:

```text
OmniReset-Arx5-OSC-Vision-v0
OmniReset-Arx5-OSC-Vision-Play-v0
```

### 4. Vision RSL-RL Config

Add or extend:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/agents/rsl_rl_vision_cfg.py
```

Suggested runner:

```text
VisionDistill_PPORunnerCfg
```

Initial training defaults:

```text
num_steps_per_env = 16
max_iterations = 40000
save_interval = 100
experiment_name = "arx5_omnireset_vision_distill"
```

Initial PPO defaults:

```text
learning_rate = 1e-4
num_learning_epochs = 4
num_mini_batches = 4
clip_param = 0.2
entropy_coef = 0.003
value_loss_coef = 1.0
gamma = 0.99
lam = 0.95
max_grad_norm = 1.0
```

Start with conservative env count:

```text
num_envs = 64 or 128
```

Raise only after confirming GPU memory and camera rendering stability.

### 5. Vision Actor-Critic Module

Add:

```text
source/uwlab_rl/uwlab_rl/rsl_rl/vision_actor_critic.py
```

Responsibilities:

- accept structured `policy` observations
- encode `external_rgb` and `wrist_rgb` with ResNet18
- encode `joint_pos` with MLP
- produce action distribution for 7D action
- compute critic value from concatenated privileged critic tensor
- expose compatible inference/export path that only needs student actor

Important implementation detail:

RSL-RL's existing MLP path assumes a flat policy tensor. Vision input should not be flattened before the network because the ResNet needs camera structure. The v1 implementation should preserve `policy` as a dictionary and keep `critic` as a tensor.

If the current `RslRlVecEnvWrapper` cannot pass nested policy dictionaries into the runner cleanly, add a minimal vision-specific wrapper or runner instead of forcing images into the existing MLP actor.

### 6. PPO + Teacher Regularization

Add:

```text
source/uwlab_rl/uwlab_rl/rsl_rl/vision_distill_ppo.py
```

or extend the existing custom PPO path if cleaner.

Responsibilities:

- load frozen teacher checkpoint
- compute teacher action from `teacher_policy`
- compute student PPO loss as usual
- add `lambda_distill * action_distill_loss`
- decay `lambda_distill`
- log `distill_loss`, `lambda_distill`, `teacher_student_action_l2`

Do not use offline demo storage. Rollouts are generated online by the student policy in Isaac.

### 7. Training Script

Add:

```text
scripts_cube_stack/07_train_vision_distill.sh
```

Initial command:

```bash
TEACHER_CKPT=logs/rsl_rl/arx5_omnireset_agent/<run>/model_<iter>.pt \
NUM_ENVS=128 \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Arx5-OSC-Vision-v0 \
  --num_envs "$NUM_ENVS" \
  --logger tensorboard \
  --headless \
  --enable_cameras \
  agent.algorithm.teacher_checkpoint="$TEACHER_CKPT" \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube \
  env.events.set_cube_stack_colors.params.insertive_object_color='[0.0,1.0,0.0]' \
  env.events.set_cube_stack_colors.params.receptive_object_color='[1.0,0.0,0.0]' \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
```

## Hydra Configuration Requirements

All important design choices must be configurable through Hydra, not hard-coded:

- camera crop size
- camera output size
- ResNet18 pretrained flag
- shared vs separate encoders
- feature dimensions
- actor/critic hidden dims
- teacher checkpoint path
- distillation loss type
- distillation coefficient schedule
- number of environments
- PPO hyperparameters

Example override style:

```bash
env.observations.policy.external_rgb.params.crop_size=400
env.observations.policy.external_rgb.params.output_size='[128,128]'
env.observations.policy.wrist_rgb.params.output_size='[128,128]'
agent.policy.vision_encoder.name=resnet18
agent.policy.vision_encoder.pretrained=false
agent.algorithm.distillation.lambda_initial=1.0
agent.algorithm.distillation.lambda_final=0.05
```

## Training Best Practices

Start small:

```text
num_envs = 64 or 128
num_steps_per_env = 16
image size = 128x128
two cameras enabled
```

Check before long training:

- verify external crop visually
- verify wrist camera orientation
- verify insertive/top cube is green and receptive/bottom cube is red
- verify policy obs contains only `joint_pos`, `external_rgb`, `wrist_rgb`
- verify teacher obs matches old state actor obs exactly
- verify teacher and student action dimensions are both 7
- verify critic obs dimension matches old critic before warm-starting critic
- verify GPU memory under one short rollout

Use staged training:

```text
Stage A: smoke test, 64 envs, 100 iterations
Stage B: short training, 128 envs, 1000-3000 iterations
Stage C: full training, increase envs only if memory allows
```

Recommended logging:

```text
episode reward
success rate
distill_loss
teacher_student_action_l2
actor entropy
value loss
approx KL
external_rgb mean/std
wrist_rgb mean/std
camera corruption count
```

## Evaluation

Use a separate vision play task:

```text
OmniReset-Arx5-OSC-Vision-Play-v0
```

Evaluation should:

- load only student actor
- not require teacher checkpoint
- not require privileged actor observations
- keep cameras enabled
- report success rate and rollout video

Initial play command shape:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-Arx5-OSC-Vision-Play-v0 \
  --checkpoint logs/rsl_rl/arx5_omnireset_vision_distill/<run>/model_<iter>.pt \
  --num_envs 16 \
  --headless \
  --enable_cameras \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

## Why Not Action Chunk in V1

A chunk output such as `K x 7` is possible, but it is not just a network output change. It requires a matching environment execution rule:

```text
student outputs K actions
env executes K internal steps
reward is accumulated
done is handled inside the chunk
next observation is returned after chunk execution
```

Without this wrapper, PPO credit assignment is inconsistent. Therefore v1 uses the current single-step 7D action. After v1 is stable, add a `ChunkedActionEnvWrapper` with `K=4` as the first chunk experiment.

## Definition of Done for V1

The first version is complete when:

- `OmniReset-Arx5-OSC-Vision-v0` starts with `--enable_cameras`
- cube stack training uses the existing workspace ranges and fixed semantic cube colors
- policy obs contains only `joint_pos`, `external_rgb`, `wrist_rgb`
- external crop is top-right 400x400 resized to 128x128
- wrist image is full-frame resized to 128x128
- student actor uses ResNet18 encoders and proprio MLP
- critic uses privileged MLP and trains online
- teacher state policy is frozen and supplies online action regularization
- training runs for at least 100 iterations without shape or camera failures
- play/export uses only the student actor

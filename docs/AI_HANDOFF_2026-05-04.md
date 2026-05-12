# UWLab ARX5 Handoff (2026-05-04)

## Current Focus

- Project: `UWLab`
- Main robot: `ARX5`
- Current work is around ARX5 peg-hole deploy / finetune alignment.

## What Changed Recently

- ARX5 deploy scene was migrated away from the previous table and aligned to the robosuite Lift-style table.
- Workspace was first expanded, then pulled back to a smaller practical region because the user found some sampled peg positions were outside reachable grasp range.
- `joint6` limit was changed to `[-pi/2, pi/2]`.
- `gripper_offset` was first changed from `0.11306` to `0.13`, and later from `0.13` to `0.145`.
- Recent work heavily focused on using existing checkpoints to finetune/deploy under the new workspace and new grasp offset instead of retraining from scratch.

## Robot Asset

- Current robot USD:
  - `source/uwlab_assets/uwlab_assets/robots/arx5/assets/arx5.usd`
- Defined in:
  - `source/uwlab_assets/uwlab_assets/robots/arx5/arx5.py`

## Important Robot Offset

- `gripper_offset` was changed from `0.13` to `0.145`.
- Current source of truth:
  - `source/uwlab_assets/uwlab_assets/robots/arx5/assets/metadata.yaml`
- Current value:

```yaml
gripper_offset:
  pos: [0.145, 0.0, 0.0]
  quat: [1.0, 0.0, 0.0, 0.0]
```

## Joint Limit Change

- `joint6` range was changed to `[-pi/2, pi/2]`.
- Relevant validation file:
  - `source/uwlab_assets/test/test_arx5_joint_limits.py`

## Deploy / Scene Notes

- Deploy table was switched to the robosuite table.
- Relevant env config:
  - `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_state_cfg.py`
- Robot / object workspace was adjusted multiple times; latest concern from user is practical deploy behavior, not broad workspace expansion.

## Table / Workspace Source Of Truth

- Table USD now used by ARX5 scene:
  - `source/uwlab_assets/uwlab_assets/props/robosuite_table/table.usd`
- Scene config points to it here:
  - `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_state_cfg.py`

### Table pose

- Current table root pose used in deploy / finetune alignment logic:

```text
table_pose = (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0)
```

### Robot pose relative to robosuite table

- Tests document the intended robosuite-aligned robot base pose:

```text
robosuite_robot_base_pose = (-0.535, -0.21, 0.8, ...)
training_robot_base_pose  = (0.0, 0.0, 0.0, ...)
```

- This is important context for why deploy/finetune workspace is in front of the robot but shifted into negative `x/y` in world coordinates.

### Current deploy workspace

- Deploy alignment currently uses:

```text
workspace_x_range = (-0.4, -0.2)
workspace_y_range = (-0.3, -0.1)
receptive_object_pose = (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
```

- Meaning:
  - `peghole` is randomized inside that workspace rectangle on reset
  - `peg` follows the corresponding translated relation after deploy alignment

- Relevant code:
  - `rl_state_cfg.py`: `DeployEvalEventCfg`
  - `mdp/events.py`: `align_deploy_scene_to_robosuite_table`

### Current finetune workspace

- Finetune reset currently uses fixed robot + randomized object workspace:

```text
robot_pose = (-0.535, -0.21, 0.8, ...)
table_pose = (0.0, 0.0, 0.799375, ...)
insertive_object_pose = (-0.30, -0.20, 0.87, 1.0, 0.0, 0.0, 0.0)
receptive_object_pose = (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
workspace_x_range = (-0.4, -0.2)
workspace_y_range = (-0.3, -0.1)
insertive_workspace_x_range = (-0.4, -0.2)
insertive_workspace_y_range = (-0.3, -0.1)
```

- If no separate insertive offset is specified, peg and peghole randomization is coupled through the reset logic in:
  - `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py`
  - class/function: `FixedRobotWorkspaceTaskPairReset`

### Tests that encode the intended scene/workspace behavior

- `source/uwlab_tasks/test/test_deploy_visual_sync.py`

Important checks already encoded there:

- deploy scene uses robosuite table
- robot aligns to robosuite Lift table edge
- deploy workspace randomization is `x in [-0.4, -0.2]`, `y in [-0.3, -0.1]`
- finetune reset uses fixed robot pose plus workspace randomization
- finetune headless training disables USD visual sync

## Latest Best Checkpoint

- Latest run inspected:
  - `logs/rsl_rl/arx5_omnireset_agent/2026-05-04_15-11-13`
- Best checkpoint from that run:
  - `logs/rsl_rl/arx5_omnireset_agent/2026-05-04_15-11-13/model_2200.pt`

### Why `model_2200.pt`

- From TensorBoard metrics in that run:
  - `task_0_success_rate` peak is around step `2194` with value about `0.6787`
  - `end_of_episode_success_rate` peak is around step `2167` with value about `0.6784`
- Checkpoint alignment:
  - `model_2200.pt` is the closest saved checkpoint to the peak
- Later checkpoints (`2300`, `2400`, `2500`) are already on the downward side.

## Latest Training Trend

- Latest run trend:
  - fast rise
  - peak near `2200`
  - then success rate declines
- This means the latest run is not monotonic; later is not better.

## Best Known Command For Latest Deploy Checkpoint

This is currently the most important known-good deploy command shape for the latest best finetune checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-State-Deploy-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/arx5_omnireset_agent/2026-05-04_15-11-13/model_2200.pt \
    agent.policy.noise_std_type=log \
    agent.policy.init_noise_std=0.1
```

Without the policy override, deploy/play can fail on actor noise tensor shape mismatch.

## Important Noise Findings

- `Mean action noise std` is the exploration std used during sampled training actions.
- For this peg insertion task, if noise grows too much:
  - success rate drops sharply
  - rollout quality degrades
  - training can collapse
- For this task, small fixed noise is much safer than letting noise drift large.

### Current understanding

- `deploy/play`: deterministic inference is fine
- `finetune`: freezing noise to a small positive value such as `0.01` or `0.02` is safer
- `noise = 0` exactly is not supported by current freeze helper and is not recommended for finetune

### Training-side noise freeze support

- Train script currently supports:

```text
--freeze_noise_std
--fixed_noise_std <positive_value>
```

- Implemented in:
  - `scripts/reinforcement_learning/rsl_rl/train.py`
  - `scripts/reinforcement_learning/rsl_rl/play_checkpoint_utils.py`

- Important:
  - exact `0.0` is rejected
  - only positive values are accepted
  - freeze is intended for non-state-dependent noise policies

## Important Play / Deploy Pitfall

### Symptom

Trying to play `model_2200.pt` with the default deploy task can fail with:

```text
size mismatch for log_std: checkpoint (7,) != current (64, 7)
```

### Root cause

- `model_2200.pt` is a finetune checkpoint.
- It was trained with:
  - `noise_std_type=log`
  - non-GSDE noise head
- But `OmniReset-Arx5-OSC-State-Deploy-Play-v0` is still registered with:
  - `Base_PPORunnerCfg`
  - which builds a GSDE-style noise structure

So the checkpoint actor noise tensor shape does not match the deploy-time policy structure.

### Related config fact

- `Base_PPORunnerCfg`:
  - `noise_std_type="gsde"`
- `Finetune_PPORunnerCfg`:
  - `noise_std_type="log"`
  - `init_noise_std=0.1`

This mismatch is the reason a finetune checkpoint may fail to load in a default deploy/play task unless CLI overrides are used.

## Correct Deploy Command For `model_2200.pt`

Use this exact command:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-State-Deploy-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/arx5_omnireset_agent/2026-05-04_15-11-13/model_2200.pt \
    agent.policy.noise_std_type=log \
    agent.policy.init_noise_std=0.1
```

Optional:

```bash
--print_actor_output
```

## Alternative Play Task

If someone wants the finetune eval environment instead of deploy camera/view settings:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-State-Finetune-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/arx5_omnireset_agent/2026-05-04_15-11-13/model_2200.pt \
    agent.policy.noise_std_type=log \
    agent.policy.init_noise_std=0.1
```

## Visualization Helper

- A helper script was added for checking current ARX5 USD and gripper offset:
  - `scripts/tools/arx5_visualize_gripper_offset.py`

## Critic Compatibility Context

There was a previous compatibility issue when trying to reuse older checkpoints after replacing the old table with the robosuite table.

### Root issue

- Old critic expected table material properties with a different raw shape / structure.
- Current code pads/truncates table material properties to a compatible dimension:
  - output dim fixed to `21`
- Relevant code:
  - `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_state_cfg.py`
  - `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/observations.py`

### Important note

- Current compatibility function repeats available material values if the table provides fewer values.
- This was added to preserve old critic checkpoint usability after the table swap.
- This is important historical context if someone later wonders why `table_material_properties` is hard-pinned to `21`.

## Checkpoint Fallback Loader Context

Custom play/train checkpoint fallback logic exists:

- file:
  - `scripts/reinforcement_learning/rsl_rl/play_checkpoint_utils.py`

It was added because:

- critic observation shapes changed across environment revisions
- strict `runner.load()` could fail even when actor weights were still reusable

Two important behaviors:

- play:
  - skips critic-only mismatches
  - rejects actor/inference mismatches
- training:
  - can warm-start actor when critic shape changed
  - can also reinitialize incompatible noise tensors when necessary

This is relevant if future work again changes privileged critic observations.

## PhysX / GPU Crash Context

The user repeatedly hit long-training Isaac/PhysX GPU failures such as:

- illegal memory access
- failed to fetch DOF velocities
- GPU kernel launch failures
- PhysX internal CUDA error
- PhysX failed to allocate GPU memory

Important conclusion from this session history:

- These were not treated as evidence that `nvidia-smi` / host driver was generally broken.
- In several cases, the likely problem was Isaac/PhysX instability during heavy long-running GPU simulation, not that the host GPU was unavailable.
- Codex sandbox visibility to GPU also caused confusing signals at times, so sandbox `nvidia-smi` failures should not be interpreted as host driver failure.

## Finetune Reset Logic

Current finetune is not using the old multi-task reset mix.

Instead, it uses:

- fixed robot pose
- fixed table pose
- peg / peghole randomized inside current workspace
- headless training path disables USD visual sync

This is encoded in:

- `FinetuneEventCfg`
- `FixedRobotWorkspaceTaskPairReset`

So if future behavior looks different, check those first before inspecting play-loop hacks.

## Success Metric Context

- Success is still based on the environment reward / progress context path.
- Rotation success is not generalized to four symmetric square-hole orientations.
- User explicitly pointed out that peg and peghole are square and ideally four rotational symmetries should count, but that change has not been implemented yet.

This is an open task / unresolved design point.

## Known Good Latest Training Run Context

For the latest best run `2026-05-04_15-11-13`:

- it used `8192` envs
- it was a finetune-style run
- its saved params confirm:
  - `noise_std_type=log`
  - `init_noise_std=0.1`
  - `schedule=fixed`
  - `entropy_coef=0.0`

This matters when reproducing or comparing later runs.

## If Work Continues

- If continuing finetune, start from `model_2200.pt`, not later checkpoints from the same run.
- If deploy behavior still looks off, first verify:
  - current `metadata.yaml` offset
  - current play task and policy noise override
  - object spawn / workspace in `rl_state_cfg.py`
- If desired long term, a cleaner fix would be to register deploy play with a finetune-compatible runner cfg instead of relying on CLI override.

## Open Ends / Things Not Finished

- No permanent registry-level fix was added yet for finetune checkpoint deploy compatibility.
  - Current workaround is CLI override of policy noise config during play.
- The square peg / square hole rotational symmetry success criterion is still not updated.
- GPU/PhysX long-run crash root cause was not fully solved at engine level; current strategy is mostly operational mitigation.

# UWLab ARX5 / Cube Stack / MuJoCo Handoff Update (2026-05-11)

## Current Focus Shift

The active work moved from peg insertion finetune/deploy to:

- cube stack policy deployment
- collecting state-policy demonstrations as HDF5
- replaying those demonstrations in Isaac
- porting the static/dynamic scene and controller to MuJoCo
- comparing Isaac replay, MuJoCo replay, and HDF5 end-effector trajectories

The user wants the Isaac environment, MuJoCo environment, robot pose, tabletop scene, cameras, and control behavior to be aligned closely enough for sim-to-sim/sim-to-real work.

## Cube Stack Environment State

Cube stack has been changed to use the same robosuite tabletop alignment/workspace family as peg insertion.

Important scene assumptions:

- robot base pose:

```text
robot_pose = (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0)
```

- table pose:

```text
table_pose = (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0)
```

- object workspace:

```text
workspace_x_range = (-0.4, -0.2)
workspace_y_range = (-0.3, -0.1)
```

Important object color decision:

- bottom/receptive cube is fixed red
- upper/insertive cube is fixed green
- cube colors should not be randomized for language/task-description clarity

Relevant code:

- `scripts_cube_stack/08_collect_state_policy_dataset.py`
- `scripts_cube_stack/09_replay_state_policy_dataset.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py`
- `source/uwlab_tasks/test/test_cube_stack_policy_dataset_collection.py`
- `source/uwlab_tasks/test/test_cube_stack_robosuite_workspace.py`

## Cube Stack Policy / Data Collection

The good cube-stack policy checkpoint used for state-policy data collection is:

```text
logs/rsl_rl/arx5_omnireset_agent/2026-05-09_18-36-45/model_8100.pt
```

Collect demonstrations with:

```bash
NUM_ENVS=4 NUM_DEMOS=50 OUTPUT_FILE=./datasets/cube_stack_state_policy.hdf5 \
bash scripts_cube_stack/08_collect_state_policy_dataset.sh \
  logs/rsl_rl/arx5_omnireset_agent/2026-05-09_18-36-45/model_8100.pt
```

Each rollout is saved as a separate HDF5 file, not under an extra `demo_0` group:

```text
./datasets/cube_stack_state_policy_demo_000000.hdf5
./datasets/cube_stack_state_policy_demo_000001.hdf5
...
```

Collection also writes a sidecar environment-camera video per rollout:

```text
./datasets/cube_stack_state_policy_demo_000000.mp4
...
```

HDF5 structure:

```text
actions                         (T, 7)
obs/table_cam                   (T, H, W, 3)
obs/wrist_cam                   (T, H, W, 3)
obs/eef_pos                     (T, 3), world frame
obs/eef_rot_6d                  (T, 6)
obs/insertive_cube_pos          (T, 3), world frame
obs/insertive_cube_quat         (T, 4)
obs/receptive_cube_pos          (T, 3), world frame
obs/receptive_cube_quat         (T, 4)
```

Important HDF5 attrs:

```text
source_env_origin
source_env_id
control_frequency_hz
env_spacing
success
table_cam_video
```

Important convention:

- `obs/eef_pos` and cube positions are recorded in Isaac world frame.
- When replaying a rollout in env0, subtract the saved `source_env_origin` first.
- This already fixed a bug where cubes replayed at the wrong location.

## Isaac Replay Command

Replay a single rollout in Isaac and save external-camera video:

```bash
bash scripts_cube_stack/09_replay_state_policy_dataset.sh \
  /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5
```

Equivalent direct command:

```bash
python scripts_cube_stack/09_replay_state_policy_dataset.py \
  --dataset_file /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --num_envs 1 \
  --env_spacing 3.0 \
  --video_path ./videos/cube_stack_replays \
  --headless
```

Replay with end-effector error metrics:

```bash
python scripts_cube_stack/09_replay_state_policy_dataset.py \
  --dataset_file /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --num_envs 1 \
  --env_spacing 3.0 \
  --video_path "" \
  --metrics_path /home/emopointer/UWLab/videos/cube_stack_replays/cube_stack_state_policy_demo_000000_isaac_eef_metrics.json \
  --headless
```

Metrics compare:

```text
Isaac replay link6 env-local position
vs
HDF5 obs/eef_pos - source_env_origin
```

Latest measured Isaac metrics on:

```text
/home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5
```

```text
count = 23
mean  = 0.012399 m
max   = 0.021486 m
final = 0.008927 m
rmse  = 0.012779 m
```

Metrics JSON:

```text
/home/emopointer/UWLab/videos/cube_stack_replays/cube_stack_state_policy_demo_000000_isaac_eef_metrics.json
```

## MuJoCo Port State

A MuJoCo ARX5 tabletop environment has been added under:

```text
mujoco_arx5/
```

Important files:

```text
mujoco_arx5/models/arx5_robot.xml
mujoco_arx5/models/arx5_robosuite_tabletop.xml
mujoco_arx5/models/arx5_tabletop_static.xml
mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml
mujoco_arx5/controllers/arx5_osc.py
mujoco_arx5/control_alignment.py
mujoco_arx5/replay_hdf5_actions.py
mujoco_arx5/README.md
```

The dynamic replay model is:

```text
mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml
```

It includes:

- ARX5 robot
- robosuite-style table
- external camera
- wrist camera
- cube objects
- ARX5 actuators
- implicitfast integrator

Important MuJoCo stability lesson:

- Do not use full visual meshes as active collision geoms for the robot.
- Visual meshes should be visual-only, with simplified collision geometry added separately.
- Earlier robot parts flew apart because adjacent visual meshes were treated as collision and/or joint refs were written in degrees while the XML compiler used radians.

Viewer command for the static/dynamic scene:

```bash
python -m mujoco.viewer \
  --mjcf mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml
```

## MuJoCo Replay Commands

Open viewer and replay HDF5 actions:

```bash
python -m mujoco_arx5.replay_hdf5_actions \
  --dataset /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --viewer \
  --no-video
```

Record MuJoCo external-camera video headlessly:

```bash
python -m mujoco_arx5.replay_hdf5_actions \
  --dataset /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --video-path /home/emopointer/UWLab/videos/mujoco_replays/cube_stack_state_policy_demo_000000_external_camera.mp4 \
  --no-real-time
```

Compute MuJoCo end-effector metrics without video:

```bash
python -m mujoco_arx5.replay_hdf5_actions \
  --dataset /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --no-video \
  --no-real-time \
  --metrics-path /home/emopointer/UWLab/videos/mujoco_replays/cube_stack_state_policy_demo_000000_mujoco_eef_metrics.json
```

Important MuJoCo rendering pitfall:

- Do not combine `--viewer` with offscreen video recording in the same process.
- On this machine, GLFW viewer plus EGL `mujoco.Renderer` crashed with:

```text
RuntimeError: Failed to make the EGL context current.
Segmentation fault
```

So:

- use `--viewer --no-video` for interactive viewing
- use no `--viewer` for headless video/metrics

## Isaac vs MuJoCo vs HDF5 EEF Error Result

Dataset used:

```text
/home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5
```

Comparison frame:

```text
env-local link6 position
```

Reference:

```text
HDF5 obs/eef_pos - source_env_origin
```

Latest metrics:

```text
Isaac replay:
  count = 23
  mean  = 0.012399 m
  max   = 0.021486 m
  final = 0.008927 m
  rmse  = 0.012779 m

MuJoCo replay:
  count = 23
  mean  = 0.057441 m
  max   = 0.098290 m
  final = 0.098290 m
  rmse  = 0.062282 m
```

Interpretation:

- Isaac replay is close to the recorded HDF5 trajectory, around 1.2 cm mean EEF error.
- MuJoCo replay is still noticeably different, around 5.7 cm mean and 9.8 cm final EEF error.
- This means the MuJoCo controller/dynamics are not fully aligned to Isaac yet.

Metrics files:

```text
/home/emopointer/UWLab/videos/cube_stack_replays/cube_stack_state_policy_demo_000000_isaac_eef_metrics.json
/home/emopointer/UWLab/videos/mujoco_replays/cube_stack_state_policy_demo_000000_mujoco_eef_metrics.json
```

## Current MuJoCo Controller Context

The MuJoCo ARX5 controller is intended to approximate the UWLab/Isaac training OSC action semantics.

Relevant file:

```text
mujoco_arx5/controllers/arx5_osc.py
```

Known controller/action facts:

- policy action dimension is 7
- first 6 dimensions are arm OSC delta command
- last dimension is gripper command
- control frequency is 10 Hz
- MuJoCo replay uses default `decimation=12` with `dt=0.008333`, giving 0.1 s control steps
- controller aligns initial MuJoCo EEF to the first HDF5 EEF by default via `--align-initial-eef`

Current MuJoCo replay first and final EEF from latest metrics:

```text
first_replay_eef = [-0.3239538669586181, -0.20253455638885498, 1.1786483526229858]
first_hdf5_eef   = [-0.32395386695861816, -0.20253455638885498, 1.1786483526229858]

final_replay_eef = [-0.46026385050554486, -0.21924073284251472, 0.964573694814302]
final_hdf5_eef   = [-0.379048228263855, -0.21281957626342773, 1.0195611715316772]
```

This confirms initial alignment works, but subsequent control/dynamics drift.

## Tests Run After Latest Replay Metrics Changes

The following targeted tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  source/uwlab_tasks/test/test_cube_stack_policy_dataset_collection.py \
  source/uwlab_tasks/test/test_mujoco_arx5_controller.py
```

Result:

```text
15 passed
```

## Sandbox / GPU Note

Isaac commands may fail inside Codex sandbox with messages like:

```text
NVML_ERROR_DRIVER_NOT_LOADED
No CUDA-capable device is detected
No device could be created
```

This does not mean the user's normal terminal/GPU is broken. In this session, the same Isaac replay command worked with non-sandbox execution and saw:

```text
NVIDIA GeForce RTX 4090
Driver Version: 570.211.01
```

For future AI assistants:

- if Isaac fails only in the sandbox, report it as sandbox GPU visibility
- do not diagnose it as a host driver failure unless the user's normal terminal also fails

## Open Next Steps

Most likely next technical work:

- reduce MuJoCo vs HDF5 EEF error by tuning MuJoCo controller gains, damping, armature, actuator model, and/or action integration semantics
- compare Isaac OSC target update semantics against `mujoco_arx5/controllers/arx5_osc.py`
- verify MuJoCo wrist/external camera world poses against Isaac-authored camera poses
- replay more HDF5 files, not only `demo_000000`, and compute aggregate Isaac/MuJoCo EEF error
- if the user asks for viewer plus video again, keep them as separate runs to avoid EGL/GLFW conflict

## More Complete File Map For The Next Assistant

This repo currently has a large dirty worktree. Do not revert unrelated files.
Many of the files below are uncommitted or locally modified as part of the
ongoing alignment work.

Core cube-stack scripts:

```text
scripts_cube_stack/_common.sh
scripts_cube_stack/08_collect_state_policy_dataset.py
scripts_cube_stack/08_collect_state_policy_dataset.sh
scripts_cube_stack/09_replay_state_policy_dataset.py
scripts_cube_stack/09_replay_state_policy_dataset.sh
docs/cube_stack_dataset_commands.md
```

Core Isaac environment/action files:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/rl_state_cfg.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/actions.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/observations.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/terminations.py
```

Core robot/table assets:

```text
source/uwlab_assets/uwlab_assets/robots/arx5/assets/arx5.urdf
source/uwlab_assets/uwlab_assets/robots/arx5/assets/arx5.usd
source/uwlab_assets/uwlab_assets/robots/arx5/assets/metadata.yaml
source/uwlab_assets/uwlab_assets/props/robosuite_table/table.usd
source/uwlab_assets/uwlab_assets/props/robosuite_table/table_only_with_external_cam.xml
```

Core MuJoCo files:

```text
mujoco_arx5/config/control_alignment.toml
mujoco_arx5/control_alignment.py
mujoco_arx5/controllers/arx5_osc.py
mujoco_arx5/models/arx5_robot.xml
mujoco_arx5/models/arx5_robosuite_tabletop.xml
mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml
mujoco_arx5/replay_hdf5_actions.py
mujoco_arx5/README.md
```

Important tests:

```text
source/uwlab_tasks/test/test_cube_stack_policy_dataset_collection.py
source/uwlab_tasks/test/test_cube_stack_robosuite_workspace.py
source/uwlab_tasks/test/test_mujoco_arx5_controller.py
source/uwlab_tasks/test/test_mujoco_arx5_robot_alignment.py
source/uwlab_tasks/test/test_mujoco_combined_scene_alignment.py
source/uwlab_tasks/test/test_mujoco_control_alignment.py
source/uwlab_tasks/test/test_mujoco_runtime_loading.py
source/uwlab_tasks/test/test_mujoco_static_scene_alignment.py
```

## Exact Wrapper Defaults

All cube-stack shell scripts source:

```text
scripts_cube_stack/_common.sh
```

That wrapper activates:

```text
conda env: env_isaaclab
cwd:       $HOME/UWLab
```

It also defines default robosuite workspace env vars:

```text
ROBOSUITE_WORKSPACE_X_MIN = -0.4
ROBOSUITE_WORKSPACE_X_MAX = -0.2
ROBOSUITE_WORKSPACE_Y_MIN = -0.3
ROBOSUITE_WORKSPACE_Y_MAX = -0.1
ROBOSUITE_TABLE_OBJECT_Z  = 0.84
ROBOSUITE_OBJECT_AIR_Z_MAX = 1.14
```

Collection wrapper defaults:

```text
NUM_ENVS=4
ENV_SPACING=3.0
NUM_DEMOS=50
OUTPUT_FILE=./datasets/cube_stack_state_policy.hdf5
MAX_STEPS_PER_DEMO=160
SEED=-1
```

Replay wrapper defaults:

```text
DATASET_FILE=/home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_*.hdf5
ENV_SPACING=3.0
VIDEO_PATH=./videos/cube_stack_replays
num_envs=1
headless=True
```

## Current Isaac Camera Setup

Deploy scene camera sensors are configured in `DeployRlStateSceneCfg`.

The cameras are not spawned by code anymore. They directly use Camera prims that
already exist in the USD hierarchy:

```text
external_camera prim_path = {ENV_REGEX_NS}/Table/external_cam/Camera
wrist_camera    prim_path = {ENV_REGEX_NS}/Robot/camera/Camera
```

Both have:

```text
height = 480
width  = 640
data_types = ["rgb"]
spawn = None
```

Important history:

- The user manually added a Camera under `/Table/external_cam`.
- The user manually added/kept a Camera under `/Robot/camera`.
- Do not add extra offset/convention transforms in the TiledCamera config unless the user explicitly asks.
- Previous bugs came from spawning an additional camera or applying an extra convention/offset, causing camera images to differ from the authored USD Camera view.

The external camera table-relative anchor used by deploy reset is:

```text
external_camera_table_relative_pose =
  (0.517, 0.327, 0.589, 0.3604, 0.2030, 0.5000, 0.7609)
```

This is relative to the current table root pose, so its world position is:

```text
(0.517, 0.327, 1.388375)
```

The MuJoCo README still records the older/static external camera anchor as:

```text
pos="0.5170 0.3270 1.3640"
quat="0.3604 0.2030 0.5000 0.7609"
fovy="42.47"
```

If comparing cameras, check which table z/reference is being used before
assuming the two numbers conflict.

## Current Isaac Action / Control Semantics

The active cube-stack task uses the built-in IsaacLab OSC action configs:

```text
Arx5OSCTrainAction -> ARX5_OSC_TRAIN
Arx5OSCEvalAction  -> ARX5_OSC_EVAL
```

Train-time OSC:

```text
body_name = link6
joint_names = joint[1-6]
target_types = ["pose_rel"]
impedance_mode = fixed
position_scale = 0.02
orientation_scale = 0.2
motion_stiffness_task = (200, 200, 200, 3, 3, 3)
motion_damping_ratio_task = (3, 3, 3, 1, 1, 1)
gravity_compensation = False
inertial_dynamics_decoupling = True
nullspace_control = none
```

Eval-time OSC:

```text
position_scale = 0.01
orientation_scale = 0.2
motion_stiffness_task = (1000, 1000, 1000, 50, 50, 50)
motion_damping_ratio_task = (1, 1, 1, 1, 1, 1)
gravity_compensation = False
inertial_dynamics_decoupling = True
nullspace_control = none
```

Important:

- The cube-stack collection policy was collected under deploy play task but the task class uses `Arx5OSCTrainAction`, not the stiff eval action.
- MuJoCo replay currently uses `mode="train"` by default to match that train action scale/gains.
- Last action dimension is binary gripper: negative closes, nonnegative opens.

## Current Deploy / Reset Semantics

Deploy eval task:

```text
task = OmniReset-Arx5-OSC-State-Deploy-Play-v0
scene class = DeployRlStateSceneCfg
events = DeployEvalEventCfg
```

Deploy reset event order includes:

```text
reset_from_reset_states
align_deploy_scene_to_robosuite_table
reject_initial_successful_resets
```

`align_deploy_scene_to_robosuite_table`:

- moves robot/table/object poses into robosuite tabletop frame
- optionally jitters robot base in x/y by `robot_xy_jitter_m`
- randomizes receptive object inside workspace if ranges are provided
- moves insertive object by the corresponding object delta
- syncs backdrop and external camera anchor
- can randomize task object colors unless disabled by caller

Deploy default in `rl_state_cfg.py` currently has:

```text
robot_xy_jitter_m = 0.02
workspace_x_range = (-0.4, -0.2)
workspace_y_range = (-0.3, -0.1)
task_object_color_range = ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))
backdrop_position_jitter_m = 0.02
backdrop_color_range = ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))
```

The cube-stack data collection and replay scripts override parts of this:

```text
robot_xy_jitter_m = 0.0
task_object_color_range = None
insertive_object_color = (0.0, 1.0, 0.0)
receptive_object_color = (1.0, 0.0, 0.0)
```

That means collected demos/replays are deterministic in robot base pose and have
semantic cube colors.

`reject_initial_successful_resets` was added because deploy sometimes reset with
peg/cube already successful. Keep this unless the user explicitly wants to
inspect initial-success cases.

## Exact Sample Dataset Details

Most recent sample file used for replay debugging:

```text
/home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5
```

HDF5 attrs:

```text
checkpoint = /home/emopointer/UWLab/logs/rsl_rl/arx5_omnireset_agent/2026-05-09_18-36-45/model_8100.pt
control_frequency_hz = 10.0
demo_index = 0
env_name = OmniReset-Arx5-OSC-State-Deploy-Play-v0
env_spacing = 3.0
num_demos = 1
num_samples = 23
schema = uwlab_cube_stack_state_policy_v2
source_env_id = 3
source_env_origin = [-1.5, 1.5, 0.0]
source_num_envs = 4
success = True
table_cam_video = datasets_test/cube_stack_state_policy_demo_000000.mp4
total_finished_rollouts = 1
```

Dataset shapes:

```text
actions                    (23, 7)          float32
obs/eef_pos                (23, 3)          float32
obs/eef_rot_6d             (23, 6)          float32
obs/insertive_cube_pos     (23, 3)          float32
obs/insertive_cube_quat    (23, 4)          float32
obs/receptive_cube_pos     (23, 3)          float32
obs/receptive_cube_quat    (23, 4)          float32
obs/table_cam              (23, 480, 640, 3) uint8
obs/wrist_cam              (23, 480, 640, 3) uint8
```

Remember:

- `source_env_origin` is not zero.
- For env0 replay comparisons, use local poses:

```text
local_pos = recorded_world_pos - source_env_origin
```

## How Isaac Replay Metrics Were Added

`scripts_cube_stack/09_replay_state_policy_dataset.py` now supports:

```text
--ee_body_name link6
--metrics_path <json path or directory>
```

The script now:

- reads `obs/eef_pos` when present
- computes recorded local EEF as `obs/eef_pos - source_env_origin`
- records replay EEF from `robot.data.body_link_pos_w[0, link6] - env_origins[0]`
- compares first `min(len(replay_trace), len(recorded_trace))` samples
- writes JSON with `mean_m`, `max_m`, `final_m`, `rmse_m`

If `--metrics_path` points to a directory or replaying multiple files, output is:

```text
<dataset_stem>_isaac_eef_metrics.json
```

## Current MuJoCo Metric Implementation

`mujoco_arx5/replay_hdf5_actions.py` supports:

```text
--metrics-path <json path>
--align-initial-eef / --no-align-initial-eef
```

By default it:

- reads HDF5 `actions`
- reads HDF5 `obs/eef_pos`
- subtracts `source_env_origin`
- resets MuJoCo to keyframe `isaac_default`
- shifts the robot root so the initial MuJoCo EEF equals the first HDF5 local EEF
- replays actions at 10 Hz with 12 substeps
- writes the same EEF metric summary as Isaac replay

This initial EEF alignment explains why MuJoCo first-frame EEF error is exactly
zero in the latest metrics, while later error grows.

## Extra Commands To Inspect Data

Extract the HDF5 environment camera to video:

```bash
python - <<'PY'
import h5py
import imageio.v2 as imageio

h5_path = "/home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5"
out = "/home/emopointer/UWLab/videos/cube_stack_replays/hdf5_table_cam_demo_000000.mp4"
with h5py.File(h5_path, "r") as f:
    fps = float(f.attrs.get("control_frequency_hz", 10.0))
    frames = f["obs/table_cam"][:]
with imageio.get_writer(out, fps=fps, codec="libx264", format="FFMPEG") as writer:
    for frame in frames:
        writer.append_data(frame)
print(out)
PY
```

Run the relevant tests after touching cube-stack/MuJoCo replay:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  source/uwlab_tasks/test/test_cube_stack_policy_dataset_collection.py \
  source/uwlab_tasks/test/test_cube_stack_robosuite_workspace.py \
  source/uwlab_tasks/test/test_mujoco_arx5_controller.py \
  source/uwlab_tasks/test/test_mujoco_arx5_robot_alignment.py \
  source/uwlab_tasks/test/test_mujoco_combined_scene_alignment.py \
  source/uwlab_tasks/test/test_mujoco_control_alignment.py \
  source/uwlab_tasks/test/test_mujoco_runtime_loading.py \
  source/uwlab_tasks/test/test_mujoco_static_scene_alignment.py
```

The shorter targeted set last run was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  source/uwlab_tasks/test/test_cube_stack_policy_dataset_collection.py \
  source/uwlab_tasks/test/test_mujoco_arx5_controller.py
```

and it passed:

```text
15 passed
```

## Known Failure Modes / Do Not Repeat

- Do not diagnose sandbox-only `nvidia-smi`/Isaac GPU errors as a broken host driver.
- Do not combine MuJoCo viewer and offscreen video recording in one process.
- Do not replay HDF5 world-frame cube/EEF positions directly into env0 without subtracting `source_env_origin`.
- Do not use robot visual meshes as active MuJoCo collision geoms.
- Do not write MuJoCo joint refs/ranges in degrees when XML compiler uses radians.
- Do not spawn extra Isaac cameras when the user-authored USD Camera prim already exists and `spawn=None` is intended.
- Do not randomize camera poses unless the user explicitly asks; the user previously corrected this.
- Do not re-randomize cube colors for the current cube-stack language-data setup; top is green, bottom is red.
- Do not revert unrelated dirty files in this repo.

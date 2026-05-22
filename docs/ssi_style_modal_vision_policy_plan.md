# SSI-Style Modal Vision Policy Plan

## Goal

Replace the current ARX5 vision actor encoder with SSI-style modal encoders adapted from the A100 `SSI-SimToReal` project.

The policy should use sim-to-real-friendly modalities rather than raw RGB as the actor input:

- depth
- bbox-derived mask
- predicted visual trajectory
- optional joint position

The new policy is single-step control. It predicts one 7D action per environment step, not an action chunk.

## Scope

This plan targets the current UWLab / Isaac RL pipeline, not the offline zarr diffusion-policy pipeline on A100.

The desired runtime flow is:

```text
Isaac cameras
  -> external RGB frame
  -> wrist RGB frame
  -> image preprocessing
  -> bbox / depth / trajectory extraction
  -> modal encoders
  -> visual tokens + trajectory tokens
  -> CLS self-attention
  -> CLS feature
  -> optionally concat joint_pos embedding
  -> actor MLP
  -> single 7D action
```

The A100 code is used as the architectural reference for these modules:

- `DepthMapEncoder`
- `MaskPatchEncoderWithoutRGB`
- `TrackPatchEmbed`
- `DepthMaskCrossAttentionFusion`
- spatial self-attention with a learnable CLS token

## Project Modification Rules

All new network structure must be implemented as explicit, reusable modules and wired through Hydra-friendly configuration. Avoid hardcoding network dimensions, layer counts, encoder names, prompts, checkpoint paths, or modality switches inside the training loop or actor class.

Required module boundaries:

```text
DepthMapEncoder
MaskPatchEncoderWithoutRGB
DepthMaskCrossAttentionFusion
TrackPatchEmbed
SpatialTokenTransformer
ProprioEncoder
SSIStyleModalEncoder
ModalVisionActorCritic
ModalExtractionService
```

Each module should have its own config block so later ablations can change one component without editing code:

```text
agent.policy.modal_encoder.depth_encoder
agent.policy.modal_encoder.mask_encoder
agent.policy.modal_encoder.track_encoder
agent.policy.modal_encoder.cross_attention
agent.policy.modal_encoder.spatial_transformer
agent.policy.use_joint_pos
agent.policy.proprio_encoder
agent.policy.actor_mlp
env.modal_extraction.depth
env.modal_extraction.bbox
env.modal_extraction.trajectory
```

Prompts must also be Hydra-configurable. Do not hardcode task prompts in Python implementation files.

Initial prompt defaults:

```text
env.modal_extraction.prompts.external: "robot, red cube, green cube"
env.modal_extraction.prompts.wrist: "red cube, green cube"
```

This is required so future tasks can switch prompts through command-line overrides, for example:

```text
env.modal_extraction.prompts.external="robot, mug, bowl"
env.modal_extraction.prompts.wrist="mug, bowl"
```

## Camera Input

The policy uses one current frame from each camera. It does not use the A100 dataset convention of two historical observation frames.

Input cameras:

```text
external_camera
wrist_camera
```

External camera preprocessing:

```text
source image: Isaac external_camera RGB
crop: top-right 400x400 pixels
intermediate image: 400x400 RGB
```

Wrist camera preprocessing:

```text
source image: Isaac wrist_camera RGB
crop: none
resize: full image -> 400x400 RGB
```

The 400x400 images are the source images for bbox detection and depth prediction. After those models run, bbox and depth are converted to 128x128 policy resolution.

## Modal Extraction

For each environment step and each view:

```text
400x400 RGB
  -> GroundingDINO / Grounded-SAM2 bbox detection
  -> DepthAnything depth prediction
  -> bbox scaled to 128x128
  -> depth resized to 128x128
  -> trajectory predictor runs on 128x128 with bbox prompts
```

External view prompt:

```text
robot, red cube, green cube
```

Wrist view prompt:

```text
red cube, green cube
```

These are defaults only. The implementation must read them from Hydra config so task-specific prompts can be changed without code edits.

Expected per-view modal tensors:

```text
depth_map:  (1, 128, 128)
bboxes:     (max_bbox_num, 5)
trajectory: (track_len, num_track_ids, 2)
```

Initial default dimensions:

```text
num_views = 2
max_bbox_num = 3
track_len = 16
num_track_ids = 32
depth_size = 128
embed_size = 256
```

The A100 IL/diffusion setup used a larger default embedding because it handled multi-task imitation learning, language conditioning, historical frames, and action chunks. This UWLab version is online RL distillation with a fixed task, one current frame, and single-step actions, so `embed_size=256` is the default complexity target.

With batch dimension `B`, the actor encoder input should be:

```text
depth_map:  (B, 2, 1, 128, 128)
bboxes:     (B, 2, 3, 5)
trajectory: (B, 2, 16, 32, 2)
```

Unlike the A100 training code, there is no observation time dimension `T=2` in this first UWLab version.

## Runtime Environment Notes

Detailed dependency, checkpoint, and setup commands are maintained in:

```text
docs/modal_vision_runtime_environment.md
```

Modal vision training runs inside the same Isaac environment used by the RSL-RL scripts:

```text
conda environment: env_isaaclab
```

The online modal extractor imports SSI-SimToReal components at runtime:

```text
GroundingDINO / Grounded-SAM2
DepthAnything
TrackTransformer trajectory predictor
BERT task embedding
```

GroundingDINO has an optional compiled extension:

```text
groundingdino._C
```

This `_C` module implements the CUDA/C++ path for `ms_deform_attn`. It is not a separate model weight; it is a compiled operator used by GroundingDINO during bbox detection.

Expected behavior:

```text
_C compiled and importable:
  detection uses the fast CUDA operator
  lower memory pressure
  preferred for training

_C missing:
  UWLab falls back to the PyTorch implementation
  results should remain semantically equivalent
  detection is slower and uses noticeably more GPU memory
  reduce NUM_ENVS and MODAL_BATCH_SIZE when debugging
```

Compile `_C` in the same Python environment used by Isaac training, because the Isaac training process imports GroundingDINO from that environment:

```bash
cd "$SSI_ROOT/Grounded_SAM_2/grounding_dino"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8

"$ISAAC_PYTHON" -m pip install -v -e .
```

Use the matching architecture when not training on A100:

```text
A100:      TORCH_CUDA_ARCH_LIST="8.0"
RTX 3090:  TORCH_CUDA_ARCH_LIST="8.6"
RTX A6000: TORCH_CUDA_ARCH_LIST="8.6"
RTX 4090:  TORCH_CUDA_ARCH_LIST="8.9"
```

Verify the extension:

```bash
export PYTHONPATH="$SSI_ROOT/Grounded_SAM_2/grounding_dino:$SSI_ROOT/Grounded_SAM_2:$PYTHONPATH"
"$ISAAC_PYTHON" -c \
"import groundingdino._C as C; print('GroundingDINO _C OK:', C)"
```

If the existing extension is named like `_C.cpython-310-*.so`, it was compiled
for Python 3.10 and will not load in Python 3.11. Rebuild it with the same
Python interpreter used by Isaac training.

If `_C` is not available, start with conservative memory settings:

```bash
NUM_ENVS=2
MODAL_BATCH_SIZE=1
DEPTH_CHUNK_SIZE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Then increase `NUM_ENVS` and `MODAL_BATCH_SIZE` only after confirming the modal extractor is stable.

## BBox To Mask

The bbox tensor should be converted into a confidence mask before encoding.

Input:

```text
bboxes: (B, V, max_bbox_num, 5)
```

Format:

```text
x1, y1, x2, y2, confidence
```

Conversion:

```text
each bbox fills a rectangle on a 128x128 mask
pixel value = bbox confidence
overlapping boxes use max confidence
padding boxes use negative coordinates and contribute 0
```

Output:

```text
mask: (B, V, 1, 128, 128)
```

## Depth Encoder

Use the A100 `DepthMapEncoder` structure:

```text
input per view: (B, 1, 128, 128)
Conv2d stride=2 -> (B, 64, 64, 64)
Conv2d kernel=8 stride=8 -> (B, 256, 8, 8)
flatten patches -> (B, 64, 256)
```

For two views:

```text
depth_tok: (B, 128, 256)
```

## Mask Encoder

Use the A100 `MaskPatchEncoderWithoutRGB` structure:

```text
input per view: (B, 1, 128, 128)
Conv2d stride=2 -> (B, 64, 64, 64)
Conv2d kernel=8 stride=8 -> (B, 256, 8, 8)
flatten patches -> (B, 64, 256)
```

For two views:

```text
mask_tok: (B, 128, 256)
```

## Depth + Mask Cross-Attention

Use the A100 `DepthMaskCrossAttentionFusion` design:

```text
query = mask_tok
key/value = depth_tok
```

Input:

```text
depth_tok: (B, 128, 256)
mask_tok:  (B, 128, 256)
```

Output:

```text
visual_tok: (B, 128, 256)
```

This keeps one fused visual token per depth/mask patch.

## Trajectory Encoder

Use the A100 `TrackPatchEmbed` idea, but remove the observation time dimension.

Input:

```text
trajectory: (B, V, 16, 32, 2)
```

Add view one-hot. With `V=2`, each trajectory point becomes:

```text
x, y, view0, view1
```

After adding view one-hot:

```text
(B, V, 16, 32, 4)
```

Flatten view into the batch for the track patch encoder:

```text
(B * V, 16, 32, 4)
```

`TrackPatchEmbed` with `patch_size=16`:

```text
track_len = 16
num_patches_per_track = 1
embed_size = 256
```

Per view output:

```text
(B * V, 1, 32, 256)
```

Merge views:

```text
traj_tok: (B, 64, 256)
```

## Spatial Fusion

The final actor encoder input tokens are:

```text
visual_tok: (B, 128, 256)
traj_tok:   (B, 64, 256)
```

Concatenate:

```text
tokens: (B, 192, 256)
```

Add one learnable CLS token:

```text
cls:     (B, 1, 256)
encoded: (B, 193, 256)
```

Run self-attention transformer:

```text
num_layers: start with 4 or 6
num_heads: 8
embed_size: 256
```

Take the CLS output:

```text
spatial_feature: (B, 256)
```

This is the direct replacement for the current ResNet-based visual actor feature.

## Actor And Critic

Actor:

```text
default:
spatial_feature: (B, 256)
  -> actor MLP
  -> action mean: (B, 7)

optional joint_pos path:
joint_pos: (B, J)
  -> proprio encoder
  -> joint_feature: (B, joint_feature_dim)
  -> concat(spatial_feature, joint_feature)
  -> actor MLP
  -> action mean: (B, 7)
```

Initial default:

```text
use_joint_pos = false
actor_input_dim = 256
```

Optional joint-pos config:

```text
use_joint_pos = true
joint_feature_dim = 32
actor_input_dim = 256 + 32 = 288
```

If enabled, the `joint_pos` embedding is concatenated after the CLS/self-attention stage. It is not inserted as a token into the visual/trajectory transformer.

The action distribution should remain compatible with the existing RSL-RL actor interface:

```text
act()
act_inference()
evaluate()
get_actions_log_prob()
update_normalization()
```

Critic:

Keep the current privileged critic path unless we explicitly decide to remove it. This keeps PPO training stable while the actor becomes perception-based.

```text
privileged critic obs -> critic MLP -> value
```

Training objective:

```text
existing online PPO distillation
PPO loss + teacher action regularization
frozen state-policy teacher
single-step 7D student action
```

## Implementation Plan

### Phase 1: Freeze The Interface

Create explicit dataclass/config entries for the new actor:

```text
policy class name
modal encoder config
per-module network config
modal extraction config
camera preprocessing config
per-view detection prompts
trajectory predictor checkpoint
depth model config
bbox detector config
```

Target files to inspect/change:

```text
source/uwlab_rl/uwlab_rl/rsl_rl/vision_actor_critic.py
source/uwlab_rl/uwlab_rl/rsl_rl/rl_cfg.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/arx5/agents/rsl_rl_vision_cfg.py
```

### Phase 2: Port Encoder Modules

Add local UWLab modules equivalent to the A100 pieces:

```text
DepthMapEncoder
MaskPatchEncoderWithoutRGB
DepthMaskCrossAttentionFusion
TrackPatchEmbed
SSIStyleModalEncoder
```

Suggested location:

```text
source/uwlab_rl/uwlab_rl/rsl_rl/modal_encoders.py
```

Unit tests should validate only tensor shapes at first:

```text
depth_map:  (B, 2, 1, 128, 128)
bboxes:     (B, 2, 3, 5)
trajectory: (B, 2, 16, 32, 2)
output:     (B, 256)
```

### Phase 3: Build Modal Actor-Critic

Add a new actor-critic class rather than mutating the current ResNet actor in place.

Suggested class:

```text
ModalVisionActorCritic
```

Responsibilities:

```text
read structured policy observations
prepare modal tensors
call SSIStyleModalEncoder
feed CLS feature to actor MLP
reuse privileged critic MLP
keep RSL-RL distribution API compatible
```

### Phase 4: Modal Observation Provider

Add an observation path that returns modal tensors, not raw RGB.

The preferred design is a batched modal extraction/cache layer immediately after Isaac camera images are available for the current environment step.

```text
Isaac camera tensors for all envs
  -> external crop / wrist resize
  -> stack all env/view images into one batch
  -> GroundingDINO bbox batch
  -> DepthAnything depth batch
  -> trajectory predictor batch
  -> cache modal tensors for this env step
  -> policy reads cached depth / bbox / trajectory
  -> policy also reads joint_pos only when use_joint_pos=true
```

Do not run GroundingDINO, DepthAnything, or trajectory prediction inside `ModalVisionActorCritic.forward()`. RSL-RL can call actor/evaluate/log-prob code multiple times per rollout update, so actor-side extraction would repeat the same expensive perception work.

For debugging:

```text
start with num_envs=1 and save modal visualization
then move to batched extraction for num_envs > 1
cache one modal result per env step
```

Performance notes:

```text
batch all envs and both views in one detector/depth/traj call when possible
keep all models loaded on GPU
use torch.no_grad()
use fp16/bfloat16 where each model supports it
use GroundingDINO-only bbox path if SAM masks are not needed
convert bbox to mask analytically on GPU
avoid CPU image round-trips
profile extraction time separately from PPO time
```

### Phase 5: Single-Env End-To-End Debug

Run one environment with visualization enabled and save:

```text
external 400x400 RGB
wrist 400x400 RGB
external bbox/mask/depth/trajectory
wrist bbox/mask/depth/trajectory
actor spatial_feature norm
joint_feature norm when use_joint_pos=true
action output
```

The first success criterion is not reward. It is that every tensor is visually and numerically plausible.

### Phase 6: PPO Training Integration

After modal extraction is correct:

```text
register new policy class in the RSL-RL runner
wire config to ModalVisionActorCritic
wire batched modal extraction/cache before actor inference
train with privileged critic
keep single-step 7D action
start with small num_envs
profile modal extraction time
```

The expected command should remain close to the current vision distillation command, with only the policy class/config changed.

### Phase 7: Performance Strategy

If online extraction is too slow, use one of these options:

```text
cache modal outputs per env step
run detector/depth/traj asynchronously
precompute modal data for offline imitation
distill from a policy trained with offline modal datasets
replace heavy detectors with lighter task-specific heads after bootstrapping
```

This is likely necessary for `num_envs=32` or higher.

## Initial Tests

Add shape-only tests:

```text
test_bbox_to_mask_shape
test_depth_encoder_shape
test_mask_encoder_shape
test_cross_attention_shape
test_track_encoder_shape
test_modal_encoder_output_shape
test_modal_actor_critic_rsl_rl_api
```

Add one integration smoke test that constructs fake observations and verifies:

```text
act() returns (B, 7)
act_inference() returns (B, 7)
evaluate() returns (B, 1)
```

## Open Questions

1. Should the language/task embedding be used?

   The A100 project has `task_emb`, but the default config there disables language token insertion into `spatial_feature`. For the fixed cube-stack task, we can omit language initially.

2. Exact batched extraction integration point in Isaac/RSL-RL.

   The current plan is to extract after env camera tensors are available and before actor inference, then cache the modal tensors for that env step. The concrete hook still needs to be selected in code.

3. Should bbox encode `robot` for wrist view?

   Current plan follows your prompt: external uses `robot, red cube, green cube`, wrist uses `red cube, green cube`. This means wrist trajectory points are object-focused only.

4. Should the trajectory predictor output exactly 32 tracks?

   The A100 encoder assumes `16 global + 16 bbox-local = 32`. The current UWLab trajectory script should be checked so its output ordering and normalization match this assumption.

## First Concrete Development Step

Implement and test the pure tensor module first:

```text
source/uwlab_rl/uwlab_rl/rsl_rl/modal_encoders.py
```

It should accept already-computed modal tensors:

```text
depth_map:  (B, 2, 1, 128, 128)
bboxes:     (B, 2, 3, 5)
trajectory: (B, 2, 16, 32, 2)
```

and return:

```text
spatial_feature: (B, 256)
```

Then wire the actor with the default no-joint path:

```text
spatial_feature: (B, 256) -> actor MLP -> action: (B, 7)
```

Keep the optional proprio encoder behind a config switch:

```text
use_joint_pos=true
joint_pos: (B, J) -> joint_feature: (B, 32)
concat(spatial_feature, joint_feature): (B, 288)
```

This isolates the network architecture from the expensive perception pipeline and makes the first code review precise.

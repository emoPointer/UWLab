# Modal Vision Runtime Environment

This document is for migrating the UWLab online modal vision policy to another
machine. It avoids machine-specific absolute paths; define the variables below
for the target server first.

```bash
export UWLAB_ROOT=/path/to/UWLab
export SSI_ROOT=/path/to/SSI-SimToReal
export ISAAC_CONDA_ENV=env_isaaclab
export ISAAC_PYTHON=/path/to/miniconda3/envs/${ISAAC_CONDA_ENV}/bin/python
export CUDA_HOME=/path/to/cuda
```

The training entry point is:

```bash
cd "$UWLAB_ROOT"
bash scripts_cube_stack/13_train_modal_vision_distill.sh
```

Do not install the full `$SSI_ROOT/requirement.txt` into the Isaac training
environment. That file belongs to the SSI imitation-learning environment and
can pin a different `torch`, `numpy`, and `opencv` stack. For UWLab modal
training, IsaacLab should keep ownership of Torch/CUDA.

## Runtime Inputs

The training script accepts these paths through environment variables:

```bash
export SSI_CONFIG="$SSI_ROOT/results/policy/0417_UWLab_delat_OSC_control_1447_seed42/config.yaml"
export TRAJECTORY_CONFIG="$UWLAB_ROOT/logs/trajectory_predict/config.yaml"
export TRAJECTORY_CKPT="$UWLAB_ROOT/logs/trajectory_predict/model_final.ckpt"
export TEACHER_CKPT="$UWLAB_ROOT/logs/rsl_rl/arx5_omnireset_agent/<run>/model_<iter>.pt"
```

The modal prompts are also runtime-configurable:

```bash
export TABLE_PROMPT="robot, red cube, green cube"
export WRIST_PROMPT="red cube, green cube"
export TASK_DESCRIPTION="Put the red block on the green block."
```

DepthAnything defaults to the smaller model for online training:

```bash
export DEPTH_ENCODER=vitb
```

## Required Model Files

DepthAnything V2:

```text
$SSI_ROOT/Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth
$SSI_ROOT/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth
```

`vitb` is the default for online training. `vitl` is optional unless you want to
compare speed or quality.

GroundingDINO / Grounded-SAM2:

```text
$SSI_ROOT/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
$SSI_ROOT/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth
$SSI_ROOT/Grounded_SAM_2/checkpoints/sam2_hiera_large.pt
```

Trajectory predictor:

```text
$UWLAB_ROOT/logs/trajectory_predict/config.yaml
$UWLAB_ROOT/logs/trajectory_predict/model_final.ckpt
```

BERT language embedding:

```text
model name: bert-base-cased
recommended cache: $SSI_ROOT/data/bert_cache
```

The online extractor defaults to `allow_bert_download=False`, so pre-cache
`bert-base-cased` before headless training. Alternatively, run once with the
Hydra observation override `allow_bert_download=true` if the server can reach
Hugging Face.

## Python Packages

The core modal-model dependencies are listed in `$UWLAB_ROOT/environment.yml`.
That file is a lightweight descriptor, not a full IsaacLab lockfile.

Important package groups:

```text
DepthAnything: timm, opencv-python, opencv-python-headless
GroundingDINO / SAM2: pycocotools, addict, yapf, supervision
Trajectory / language: transformers, huggingface-hub, einops
Config / video tools: omegaconf, hydra-core, imageio, imageio-ffmpeg
```

Repair/install modal dependencies in the active Isaac environment:

```bash
"$ISAAC_PYTHON" -m pip install \
    numpy==1.26.0 \
    opencv-python==4.11.0.86 \
    opencv-python-headless==4.11.0.86 \
    timm \
    pycocotools \
    addict \
    yapf \
    supervision \
    einops \
    transformers \
    huggingface-hub \
    omegaconf \
    hydra-core \
    imageio \
    imageio-ffmpeg
```

Do not add `torch`, `torchvision`, or CUDA wheels with this command unless the
IsaacLab environment itself is being rebuilt.

## GroundingDINO CUDA Extension

GroundingDINO has an optional compiled extension:

```text
groundingdino._C
```

This is a C++/CUDA operator for `ms_deform_attn`, not a checkpoint. With `_C`,
bbox detection is faster and uses less GPU memory. Without `_C`, UWLab falls
back to the PyTorch implementation; the detection output should stay
semantically equivalent, but online training is slower and uses more memory.

Compile it inside the same Python environment used by Isaac training:

```bash
cd "$SSI_ROOT/Grounded_SAM_2/grounding_dino"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8

"$ISAAC_PYTHON" -m pip install -v -e .
```

Choose `TORCH_CUDA_ARCH_LIST` for the target GPU:

```text
A100:      8.0
RTX 3090:  8.6
RTX A6000: 8.6
RTX 4090:  8.9
```

Verify:

```bash
export PYTHONPATH="$SSI_ROOT/Grounded_SAM_2/grounding_dino:$SSI_ROOT/Grounded_SAM_2:$PYTHONPATH"
"$ISAAC_PYTHON" -c "import groundingdino._C as C; print('GroundingDINO _C OK:', C)"
```

If this fails with `ModuleNotFoundError`, `_C` is not available to the active
Isaac Python. A file named like `_C.cpython-310-*.so` does not work with Python
3.11; rebuild `_C` with the exact `$ISAAC_PYTHON` used for training.

## Validation

Check core imports:

```bash
MPLCONFIGDIR=/tmp/matplotlib "$ISAAC_PYTHON" -c \
"import torch, cv2, timm, transformers, supervision, pycocotools; print(torch.__version__, cv2.__version__)"
```

Check required local files:

```bash
test -f "$SSI_ROOT/Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth"
test -f "$SSI_ROOT/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth"
test -f "$SSI_ROOT/Grounded_SAM_2/checkpoints/sam2_hiera_large.pt"
test -f "$TRAJECTORY_CKPT"
```

Start with conservative memory settings when validating a new server:

```bash
cd "$UWLAB_ROOT"

NUM_ENVS=2 \
MODAL_BATCH_SIZE=1 \
DEPTH_CHUNK_SIZE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts_cube_stack/13_train_modal_vision_distill.sh
```

After the perception stack is confirmed, increase `NUM_ENVS`,
`MODAL_BATCH_SIZE`, and `DEPTH_CHUNK_SIZE` according to available GPU memory.

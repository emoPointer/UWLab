# Cube Stack Dataset Commands

## Collect State-Policy Data

Use the trained cube-stack state-policy checkpoint:

```bash
NUM_ENVS=4 NUM_DEMOS=50 OUTPUT_FILE=./datasets/cube_stack_state_policy.hdf5 \
bash scripts_cube_stack/08_collect_state_policy_dataset.sh \
  logs/rsl_rl/arx5_omnireset_agent/2026-05-09_18-36-45/model_8100.pt
```

Each successful rollout is saved as one HDF5 file:

```text
./datasets/cube_stack_state_policy_demo_000000.hdf5
./datasets/cube_stack_state_policy_demo_000001.hdf5
...
```

The collection script also saves a sidecar environment-camera video for each rollout:

```text
./datasets/cube_stack_state_policy_demo_000000.mp4
./datasets/cube_stack_state_policy_demo_000001.mp4
...
```

Common overrides:

```bash
NUM_ENVS=8
NUM_DEMOS=200
ENV_SPACING=3.0
MAX_STEPS_PER_DEMO=160
OUTPUT_FILE=./datasets/cube_stack_state_policy.hdf5
```

## Replay Collected Data

Replay all collected HDF5 files and save videos:

```bash
VIDEO_PATH=./videos/cube_stack_replays \
bash scripts_cube_stack/09_replay_state_policy_dataset.sh \
  "/home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_*.hdf5"
```

Replay a single rollout:

```bash
VIDEO_PATH=./videos/cube_stack_replays \
bash scripts_cube_stack/09_replay_state_policy_dataset.sh \
  /home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_000000.hdf5
```

Replay output videos are saved under `VIDEO_PATH`, for example:

```text
./videos/cube_stack_replays/cube_stack_state_policy_demo_000000_external_replay.mp4
```

Notes:

- Collection uses the same control frequency as the environment.
- HDF5 files store raw actor actions, two camera streams, EEF pose, and cube poses.
- Replay is deterministic with robot-base jitter disabled.

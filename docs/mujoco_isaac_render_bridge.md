# MuJoCo Renderer With Isaac Physics

This bridge keeps Isaac/PhysX as the only physics authority and mirrors one
Isaac environment into a MuJoCo scene for rendering.

## Ownership

- Isaac owns simulation, contacts, resets, object motion, policy stepping, and done signals.
- MuJoCo owns only visual kinematics and RGB rendering.
- The bridge calls `mj_forward`, not `mj_step`.

## Added Modules

- `mujoco_arx5/isaac_render_bridge/state.py`: typed render-state snapshot.
- `mujoco_arx5/isaac_render_bridge/state_extractors.py`: reads state from an IsaacLab env.
- `mujoco_arx5/isaac_render_bridge/mujoco_mirror.py`: writes state into `MjModel`/`MjData`.
- `mujoco_arx5/isaac_render_bridge/renderer.py`: offscreen RGB and MP4 helpers.
- `mujoco_arx5/isaac_render_bridge/session.py`: reusable Isaac-physics/MuJoCo-render session.
- `scripts_mujoco_isaac/play_with_mujoco_renderer.py`: independent play script.

## Current Scope

The default MuJoCo XML mirrors the ARX5 robot plus cube-shaped task objects:

- `insertive_object` -> `insertive_cube`
- `receptive_object` -> `receptive_cube`
- `joint1`..`joint8` -> same-name MuJoCo joints

For peg/peghole visual parity, add a separate MuJoCo render XML with peg and
peghole visual bodies, then pass its body mapping through
`--mujoco_insertive_body` and `--mujoco_receptive_body`.

## Example

```bash
python scripts_mujoco_isaac/play_with_mujoco_renderer.py \
  logs/rsl_rl/arx5_omnireset_agent/2026-05-09_18-36-45/model_8100.pt \
  --task <TASK_NAME> \
  --record_mujoco_video \
  --mujoco_video_path logs/mujoco_isaac_bridge/model_8100_external.mp4 \
  env.scene.insertive_object=cube \
  env.scene.receptive_object=cube
```

The script mirrors env 0. Use `--max_steps` or `--stop_on_done` for bounded
recordings.

# ARX5 MuJoCo Alignment

This directory is for migrating the current Isaac ARX5 tabletop setup into MuJoCo in stages.

## Stage 1: Static Scene

`models/arx5_tabletop_static.xml` contains only the static tabletop scene plus reference task objects. It intentionally does not include the ARX5 robot yet.

Aligned values copied from the current Isaac setup:

- Table: robosuite table-only geometry, source body pose `0 0 0.775`, table top at `z=0.800`.
- Workspace: `x=[-0.4, -0.2]`, `y=[-0.3, -0.1]`.
- Robot reference pose for the next stage: `(-0.535, -0.21, 0.8, 1, 0, 0, 0)`.
- External camera anchor: `pos="0.5170 0.3270 1.3640"`, `quat="0.3604 0.2030 0.5000 0.7609"`, `fovy="42.47"`.
- Backdrop curtains: current Isaac deploy curtain positions, sizes converted to MuJoCo half extents.
- Cube-stack semantic colors: receptive/bottom cube red, insertive/top cube green.

## Stage 2: Robot

`models/arx5_robot.xml` contains the ARX5 kinematic tree copied from the current URDF/Isaac asset:

- Root pose: `(-0.535, -0.21, 0.8, 1, 0, 0, 0)`.
- Arm joints: `joint1..joint6`, with the current `joint6` range `[-1.5708, 1.5708]`.
- Gripper slide joints: `joint7/joint8`, range `[0.0, 0.044]`.
- Default keyframe: `joint1=0`, `joint2=1`, `joint3=1`, `joint4=0`, `joint5=0`, `joint6=0`, `joint7=0.02`, `joint8=0.02`.
- Grasp point site: `link6` local position `(0.145, 0, 0)`.
- Wrist camera is under the existing `camera` link with identity camera offset.
- Camera visual meshes from the URDF are represented as simple MuJoCo box geoms because MuJoCo 3.8.0 cannot decode the current `.glb` mesh files directly.
- Robot collision follows the SSI-SimToReal robosuite ARX5 convention: visual meshes are visual-only, and simplified box collision primitives are added per link/finger.
- The robot XML uses `integrator="implicitfast"` because the Isaac-aligned gripper
  position actuator gains are too stiff for MuJoCo's default Euler integrator at
  the current `1/120s` step.

## Stage 3: Control Reference

`config/control_alignment.toml` and `control_alignment.py` record the current Isaac control semantics:

- Physics dt: `1/120`.
- Decimation: `12`.
- Policy/control frequency: `10 Hz`.
- Eval OSC scale: xyz `0.01`, axis-angle `0.2`.
- Train OSC scale: xyz `0.02`, axis-angle `0.2`.
- Arm actuator interface: six torque motors, each clipped to `[-50, 50]`.
- Gripper action: one scalar, negative closes to `0.002`, nonnegative opens to `0.044`.

`controllers/arx5_osc.py` implements the MuJoCo-side ARX5 OSC controller against
`models/arx5_robot.xml`:

- Input action: `[dx, dy, dz, dax, day, daz, gripper]`.
- Default mode: `train`, matching the current UWLab training action scale/gains.
- EE body: `link6`.
- Arm outputs: `joint1_torque..joint6_torque`.
- Gripper outputs: `joint7_position`, `joint8_position`.
- OSC form: relative pose target, zero desired EE velocity, MuJoCo mass-matrix
  operational-space inertia decoupling, no gravity compensation, no nullspace
  command.

Minimal use:

```python
import mujoco
from mujoco_arx5.controllers import Arx5OperationalSpaceController

model = mujoco.MjModel.from_xml_path("mujoco_arx5/models/arx5_robot.xml")
data = mujoco.MjData(model)
controller = Arx5OperationalSpaceController(model, mode="train")
controller.reset(data)

action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
controller.apply_action(data, action)
mujoco.mj_step(model, data)
```

Replay one collected HDF5 action rollout on the MuJoCo robot:

```bash
python -m mujoco_arx5.replay_hdf5_actions \
  --dataset /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --viewer \
  --no-video
```

The script updates the relative target once per recorded 10 Hz action and then
tracks that held target for 12 MuJoCo physics substeps.  By default it uses
`models/arx5_robosuite_tabletop_dynamic.xml`, which contains the aligned robot
root pose, robosuite table, `external_camera`, and `wrist_camera`.  It also
reads the first recorded HDF5 EEF position and shifts the robot root by the
recorded base jitter so the MuJoCo initial EEF pose matches the collected
rollout.  Disable that with `--no-align-initial-eef` when debugging the nominal
base pose.

Record the environment camera video in headless mode:

```bash
python -m mujoco_arx5.replay_hdf5_actions \
  --dataset /home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5 \
  --video-path /home/emopointer/UWLab/videos/mujoco_replays/replay.mp4
```

Do not combine `--viewer` with video recording in one process: MuJoCo's GLFW
viewer and EGL offscreen renderer can conflict on this workstation. Use
`--viewer --no-video` for interactive viewing, and run a separate no-viewer
command to save MP4s.

Validation includes XML/config structure tests and MuJoCo runtime loading tests.

## Combined Viewer Scene

Open the combined UWLab-aligned scene with:

```bash
python -m mujoco.viewer --mjcf mujoco_arx5/models/arx5_robosuite_tabletop.xml
```

This scene uses the current UWLab deploy/data collection poses:

- Robot root: `(-0.535, -0.21, 0.8)`.
- Table source body: `(0, 0, 0.775)`, matching the converted Isaac tabletop height.
- Receptive cube: `(-0.30, -0.20, 0.84)`.
- Insertive cube: `(-0.30, -0.20, 0.87)`.

For this first viewer scene, robot visual meshes remain visual-only, robosuite-style simplified box collision geoms are present, actuators are omitted, and gravity is disabled. The goal is to inspect the current UWLab visual/collision alignment without uncontrolled robot dynamics.

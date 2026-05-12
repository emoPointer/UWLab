## [Unreleased] - 2026-05-11
### Features
- Started ARX5 Isaac-to-MuJoCo migration with a static robosuite tabletop scene XML.
- Added ARX5 MuJoCo robot alignment XML and a control-alignment config for current Isaac OSC/gripper semantics.
- Added a combined MuJoCo viewer scene with the current UWLab robot/table/workspace/camera poses.
- Added a MuJoCo ARX5 operational-space controller matching the current UWLab 7D policy action contract.
- Added a MuJoCo HDF5 action replay entry point for collected cube-stack state-policy rollouts.
- Added a dynamic robosuite tabletop replay scene with aligned robot, table, external camera, wrist camera, and actuators.

### Design Rationale
- The migration is staged: static scene first, robot import/alignment second, control alignment third. This keeps coordinate and asset differences isolated.
- The MuJoCo controller uses MuJoCo Jacobians and mass matrices to mirror IsaacLab OSC inertial decoupling while keeping the API independent from IsaacLab.
- The dynamic ARX5 robot XML uses MuJoCo `implicitfast` integration so the stiff gripper position actuators remain stable at the Isaac-aligned 1/120s physics step.
- HDF5 action replay aligns the robot root to the recorded first-frame EEF position so datasets collected with robot-base jitter start from the same pose.

### Notes & Caveats
- The combined viewer scene is still static; the controller targets `mujoco_arx5/models/arx5_robot.xml` until a dynamic tabletop task scene is assembled.

# UWLab

UWLab 是基于 Isaac Lab 的机器人学习实验仓库，当前主要任务线是 **OmniReset**（抓取采样 → 重置状态 → RL 训练 → sim2real 蒸馏）。支持的机器人包括 UR5e (Robotiq gripper) 和 ARX5（6-DOF + 平行夹爪）。

## 环境

- Conda env: `env_isaaclab`（位于 `~/miniconda3/envs/env_isaaclab/`）
- 激活方式：`source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
- Isaac Lab 路径：`/home/emopointer/UWLab/_isaaclab/IsaacLab/`
- ARX5 USD/URDF 存放位置：`source/uwlab_assets/uwlab_assets/robots/arx5/assets/`（已从 rl_SSI_STR 拷贝到项目内）

## 关键目录

- `source/uwlab_assets/uwlab_assets/robots/` — 机器人配置（`ur5e_robotiq_gripper/`、`arx5/` 等）
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/` — OmniReset 任务实现
  - `mdp/actions/task_space_actions.py` — 自定义 `RelCartesianOSCAction`（目前只给 UR5e 用）
  - `config/<robot>/` — 各机器人的任务配置
- `scripts/tools/` — 工具脚本（replay、FK 验证等）

## OSC 控制器现状

OmniReset 里有两套 OSC 路径：

- **自定义 `RelCartesianOSCAction`**（`mdp/actions/task_space_actions.py`）：
  - `τ = J^T · (Kp·pose_error + Kd·(−ee_vel))`，无惯量解耦
  - 依赖 `uwlab_assets.robots.ur5e_robotiq_gripper.kinematics.compute_jacobian_analytical`
  - **硬编码 import 了 UR5e 的 analytical Jacobian**，所以 ARX5 无法直接用
- **Isaac Lab 官方 `OperationalSpaceControllerActionCfg`**：ARX5 当前用的就是这个（`config/arx5/actions.py`），带 inertial decoupling + gravity compensation

要让 ARX5 用自定义 OSC，缺的是：
1. `robots/arx5/kinematics.py`（Rodrigues-based FK/Jacobian，支持 ARX5 的混合关节轴 Z/Y/Y/Y/Z/X 和 joint3/joint6 的 π-flip）
2. `assets/metadata.yaml` 补 `calibrated_joints.{xyz,rpy,axis}` 和 `link_inertials` 两块
3. `RelCartesianOSCAction` 参数化，允许注入 `jacobian_fn` 而不是 hardcode UR5e 导入

## sim2real 对齐经验

### ⚠️ ARX5 真机 `eef_qpos` 不是 base_link 下的绝对位姿

**真机的 `eef_qpos[:3]` 是 link6 相对于"上电时刻" link6 位姿的 delta**，而不是 `base_link` 坐标系下的绝对位姿。`eef_qpos[3:6]` 同理（相对旋转的 RPY）。

#### 来源（实测验证）

用 `scripts/tools/replay_arx5_fk.py` replay `/home/emopointer/UWLab/0.hdf5`（100 帧真机数据）得到：
- 旋转误差均值 0.2°，最大 1.3° — 几乎完美
- 平移误差在各帧之间**近乎恒定**，均值 `(96.48, 0.57, 154.70) mm`，帧间波动 < 3mm

再用 `scripts/tools/arx5_fk_at_zero.py` 验证 `q_arm = [0,0,0,0,0,0]`：
```
link6 in base_link frame: pos = (97.7, -0.5, 156.5) mm, rpy = (0, 0, 0)
```

这和 replay 得到的常量偏移只差 `(1.2, -0.5, 2.5) mm`（因为 hdf5 第一帧 qpos 也不是精确 0，接近 0）。所以确认：
```
eef_qpos_real(t)  ≈  T_link6(q(t))  ⊖  T_link6(q_boot)
```

#### 含义

1. **ARX5 URDF 几何本身是精准的**：去掉恒定偏移后残差 <3mm，不需要重新标定 `calibrated_joints.xyz/rpy`
2. **做 sim2real FK 对比时必须用 delta 对齐**：两边都减掉各自的初始 pose 再比较，否则会看到一个 ~160mm 的假性误差
3. **`RelCartesianOSCAction` 天然匹配**：因为它本来就是 delta 控制，和真机数据的定义一致
4. **别混淆 `link6` 和 gripper TCP**：真机 `eef_qpos` 是 link6，不是 `gripper_offset` 之后的 TCP。metadata.yaml 里 `gripper_offset = (0.11306, 0, 0)` 只在做 TCP-level 任务时才加

#### 踩坑记录

- 第一次 replay 结果显示 `(96, 0, 154) mm` 的恒定偏移，差点误判成 URDF 错误
- 复查发现旋转几乎零误差 + 平移帧间波动极小 → 不是几何错误而是**帧定义**错误
- 进一步验证 `link6(q=0) ≈ (97.7, -0.5, 156.5) mm` 与偏移匹配 → 确认真机 `eef_qpos` 是相对上电 pose 的 delta

### IMPLICIT_ARX5 的 zero-stiffness actuator 注意事项

`source/uwlab_assets/uwlab_assets/robots/arx5/arx5.py` 里 `IMPLICIT_ARX5` 的 arm actuator 是 `stiffness=0.0, damping=0.0`。这意味着：
- 用 `write_joint_state_to_sim` + `sim.step` 来做"伪 FK"时，重力会把关节从写入的目标拖走
- **FK 验证脚本必须关重力**：`SimulationCfg(gravity=(0.0, 0.0, 0.0))`，否则会看到 ~1-2 mm 的假性残差
- `scripts/tools/replay_arx5_fk.py` 和 `scripts/tools/arx5_fk_at_zero.py` 都已经这么做了

## 工具脚本

- `scripts/tools/replay_arx5_fk.py` — 读 hdf5 的 `(qpos, eef_qpos)`，在 Isaac Sim 里 replay joint_pos，对比 FK vs 记录的 ee_pose。支持 `--headless` / GUI 模式，支持 `--frame_delay` 控制可视化速度。
- `scripts/tools/arx5_fk_at_zero.py` — 把 arm 关节设为 0，打印 link6 在 base_link 下的位姿。用来验证"真机 eef_qpos 原点"假设。

## Code Review

按全局 `~/.claude/CLAUDE.md` 的 Tier 1/Tier 2 规则走。对于 sim2real 相关的控制器/FK 改动，务必把"帧定义"和"stiffness=0 下的 gravity drift"两个上下文在 context summary 里带上，否则 Codex 看不到这层约束。

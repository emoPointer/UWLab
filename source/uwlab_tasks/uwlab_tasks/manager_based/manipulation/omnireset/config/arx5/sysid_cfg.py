# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene and manager-based env config for ARX5 system identification (CMA-ES).

Reuses the same robot and RelCartesianOSCAction as RL.  Sysid scripts use the
registered gym env so the in-env OSC is used (no duplicate controller).

Mirrors the UR5e sysid_cfg structure: minimal scene (robot + ground + light),
minimal MDP (joint_pos obs, no reward, long timeout), and decimation=1 so
every physics step = one env.step().
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.arx5 import EXPLICIT_ARX5

from ... import mdp as task_mdp
from .actions import Arx5SysidOSCAction

# Default simulation timestep for sysid (500 Hz, matches MuJoCo and real robot)
SYSID_SIM_DT = 1.0 / 500.0


@configclass
class SysidSceneCfg(InteractiveSceneCfg):
    """Scene for system identification: robot + ground + light, no objects."""

    robot = EXPLICIT_ARX5.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class SysidObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SysidRewardsCfg:
    pass


@configclass
class SysidTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class SysidEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env for sysid: same scene + RelCartesianOSC as RL, decimation=1."""

    scene: SysidSceneCfg = SysidSceneCfg(num_envs=512, env_spacing=2.0)
    actions: Arx5SysidOSCAction = Arx5SysidOSCAction()
    observations: SysidObservationsCfg = SysidObservationsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1
        self.episode_length_s = 99999.0
        self.sim.dt = SYSID_SIM_DT

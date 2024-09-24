# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher()
from omni.isaac.lab_tasks.manager_based.manipulation.lift import franka_lift_env
from omni.isaac.lab_tasks.manager_based.manipulation.stack import franka_stack_env


##
# Inverse Kinematics - Relative Pose Control
##

base_dir = os.path.dirname(__file__)
bc_json_path = os.path.join(base_dir, '..', 'exps', 'templates', 'bc.json')

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.manipulation.lift:FrankaCubeLiftEnv",
    kwargs={
        "env_cfg_entry_point": franka_lift_env.FrankaCubeLiftEnv,
        "robomimic_bc_cfg_entry_point": bc_json_path,
    },
    disable_env_checker=True,
)

gym.register(
    id="PickPlace_D0",
    entry_point="omni.isaac.lab_tasks.manager_based.manipulation.lift:FrankaCubeLiftEnv",
    kwargs={
        "env_cfg_entry_point": franka_lift_env.FrankaCubeLiftEnv,
        "robomimic_bc_cfg_entry_point": bc_json_path,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.manipulation.stack:FrankaCubeStackEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_env.FrankaCubeStackEnv,
        "robomimic_bc_cfg_entry_point": bc_json_path,
    },
    disable_env_checker=True,
)

gym.register(
    id="Stack_D0",
    entry_point="omni.isaac.lab_tasks.manager_based.manipulation.stack:FrankaCubeStackEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_env.FrankaCubeStackEnv,
        "robomimic_bc_cfg_entry_point": bc_json_path,
    },
    disable_env_checker=True,
)

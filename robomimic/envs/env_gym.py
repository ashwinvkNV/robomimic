"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

import gymnasium as gym
try:
    import d4rl
except:
    print("WARNING: could not load d4rl environments!")

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
from omni.isaac.lab.utils.io import load_pickle
import os
import torch

from omni.isaac.lab_tasks.manager_based.manipulation.stack import mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm


class EnvGym(EB.EnvBase):
    """Wrapper class for gym"""
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): ignored - gym envs always support on-screen rendering

            render_offscreen (bool): ignored - gym envs always support off-screen rendering

            use_image_obs (bool): ignored - gym envs don't typically use images

            postprocess_visual_obs (bool): ignored - gym envs don't typically use images
        """
        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None
        # TODO: Automatically load the config as per env
        # env_cfg = load_pickle(os.path.join("/home/ashwin/git_cloned/IsaacLab-Internal/logs/robomimic/Isaac-Lift-Cube-Franka-IK-Rel-v0/params", "env.pkl"))
        env_cfg = load_pickle(os.path.join("/home/ashwin/git_cloned/IsaacLab-Internal/logs/robomimic/Isaac-Stack-Cube-Franka-IK-Rel-v0/params", "env.pkl"))
        
        # TODO: Enable randomization configuration during data generation
        env_cfg.events.randomize_franka_joint_state = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.4, 0.4),
                "velocity_range": (0, 0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        randomization_range_top = 0.05
        randomization_range_bottom = -0.05
        env_cfg.events.randomize_cube_1_position.params["pose_range"]["x"] = (randomization_range_bottom, randomization_range_top)
        env_cfg.events.randomize_cube_1_position.params["pose_range"]["y"] = (randomization_range_bottom, randomization_range_top)
        env_cfg.events.randomize_cube_2_position.params["pose_range"]["x"] = (randomization_range_bottom, randomization_range_top)
        env_cfg.events.randomize_cube_2_position.params["pose_range"]["y"] = (randomization_range_bottom, randomization_range_top)
        env_cfg.events.randomize_cube_3_position.params["pose_range"]["x"] = (randomization_range_bottom, randomization_range_top)
        env_cfg.events.randomize_cube_3_position.params["pose_range"]["y"] = (randomization_range_bottom, randomization_range_top)

        self.env = gym.make(env_name, cfg=env_cfg, **kwargs)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        if action.dim() == 1 and action.size(0) == 7:
            action = action.unsqueeze(0)  # Reshape to [1, 7]
        else:
            # Optionally, handle cases where the action is not a 1D tensor of size 7
            raise ValueError("Action must be a 1D tensor with 7 elements")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = (terminated | truncated)
        return self.get_observation(obs), reward, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self._current_obs = self.env.reset()
        self._current_reward = None
        self._current_done = None
        return self.get_observation(self._current_obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        if hasattr(self.env.unwrapped.sim, "set_state_from_flattened"):
            self.env.unwrapped.sim.set_state_from_flattened(state["states"])
            self.env.unwrapped.sim.forward()
            return { "flat" : self.env.unwrapped._get_obs() }
        elif hasattr(self.env, "reset_to"):
            self.env.reset_to(state)
            return
        else:
            raise NotImplementedError

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            return self.env.render(mode=mode, **kwargs)
        if mode == "rgb_array":
            return self.env.render(mode="rgb_array", height=height, width=width)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            assert self._current_obs is not None
            obs = self._current_obs
        return { "flat" : np.copy(obs) }

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        assert self._current_obs is not None
        assert self._current_obs is not None
        obs = self._current_obs
        return dict(states=obs)

    def get_reward(self):
        """
        Get current reward.
        """
        assert self._current_reward is not None
        return self._current_reward

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        assert self._current_done is not None
        return self._current_done

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        if hasattr(self.env.unwrapped, "_check_success"):
            return self.env.unwrapped._check_success()
        
        # TODO: Find a better way to check for scene is lift or stack scene
        
        if "object" in self.env.scene.keys():
            lifted = self.env.scene["object"].data.root_pos_w[:, 2].cpu().numpy()[0] > 0.1
        else:
            xy_threshold = 0.03
            height_threshold = 0.005
            height_diff = 0.0468
            cube_1: RigidObject = self.env.scene["cube_1"]
            cube_2: RigidObject = self.env.scene["cube_2"]
            cube_3: RigidObject = self.env.scene["cube_3"]

            pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
            pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

            # Compute cube position difference in x-y plane
            xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
            xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

            # Compute cube height difference
            h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
            h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

            stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
            stacked = torch.logical_and(torch.norm(h_dist_c12 - height_diff) < height_threshold, stacked)
            stacked = torch.logical_and(torch.norm(h_dist_c23 - height_diff) < height_threshold, stacked)
            # return stacked
            return { "task" : torch.any(stacked).item() }

        return { "task" : lifted }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_space.shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.GYM_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        render=None, 
        render_offscreen=None, 
        use_image_obs=None, 
        use_depth_obs=None, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvGym instance)
        """

        # make sure to initialize obs utils so it knows which modalities are image modalities.
        # For currently supported gym tasks, there are no image observations.
        obs_modality_specs = {
            "obs": {
                "low_dim": ["flat"],
                "rgb": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

"""Vectorized Mujoco environment wrapper using gymnasium."""
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


def _make_env(env_name: str, seed: int, render_mode: str | None = None):
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init


class MujocoEnv:
    """Parallel Mujoco environments using SyncVectorEnv."""

    def __init__(self, env_name: str, num_envs: int, seed: int = 42):
        self.num_envs = num_envs
        self.env_name = env_name

        fns = [_make_env(env_name, seed + i) for i in range(num_envs)]
        self.envs = SyncVectorEnv(fns)

        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space
        self.obs_dim: int = int(np.prod(obs_space.shape))
        self.act_dim: int = int(np.prod(act_space.shape))

    def reset(self) -> np.ndarray:
        obs, _ = self.envs.reset()
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        obs, rewards, terminated, truncated, infos = self.envs.step(actions)
        dones = terminated | truncated
        return (
            obs.astype(np.float32),
            rewards.astype(np.float32),
            dones,
            infos,
        )

    def close(self):
        self.envs.close()

    # ------------------------------------------------------------------
    # Single-env rendering helpers for video collection
    # ------------------------------------------------------------------

    def make_render_env(self, seed: int = 0):
        """Return a single env with rgb_array rendering for video capture."""
        env = gym.make(self.env_name, render_mode="rgb_array")
        env.reset(seed=seed)
        return env

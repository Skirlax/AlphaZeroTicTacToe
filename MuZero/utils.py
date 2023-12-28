import numpy as np
import torch as th
from PIL import Image


def add_actions_to_obs(observations: th.Tensor, actions: th.Tensor,dim=0):
    return th.cat((observations, actions), dim=dim)


def match_action_with_obs(observations: th.Tensor, action: int):
    action = th.full((1, observations.shape[1], observations.shape[2]), action, dtype=th.float32,
                     device=observations.device)
    return add_actions_to_obs(observations, action)


def match_action_with_obs_batch(observation_batch: th.Tensor, action_batch: list[int]):
    tensors = [th.full((1, 1,observation_batch.shape[2],observation_batch.shape[3]), action,
                       dtype=th.float32, device=observation_batch.device) for action in action_batch]
    actions = th.cat(tensors, dim=0)
    return add_actions_to_obs(observation_batch, actions,dim=1)


def resize_obs(observations: np.ndarray, size: tuple[int, int]):
    obs = Image.fromarray(observations)
    obs = obs.resize(size)
    return np.array(obs)


def scale_state(state: np.ndarray):
    # scales the given state to be between 0 and 1
    max_val = np.max(state)
    min_val = np.min(state)
    return (state - min_val) / (max_val - min_val)


def scale_action(action: int, num_actions: int):
    return action / (num_actions - 1)


def scale_reward_value(value: th.Tensor, e: float = 0.001):
    if isinstance(value, float) or isinstance(value,np.float32):
        scaled_v = np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1 + value * e)
        return np.array([scaled_v])
    return th.sign(value) * (th.sqrt(th.abs(value) + 1) - 1 + value * e)

import torch as th


def add_actions_to_obs(observations: th.Tensor, actions: th.Tensor):
    return th.cat((observations, actions), dim=2)

import numpy as np
import optuna
import torch as th
from PIL import Image

from AlphaZero.Arena.players import NetPlayer
from AlphaZero.constants import SAMPLE_MZ_ARGS
from AlphaZero.utils import DotDict



def add_actions_to_obs(observations: th.Tensor, actions: th.Tensor, dim=0):
    return th.cat((observations, actions), dim=dim)


def match_action_with_obs(observations: th.Tensor, action: int):
    action = th.full((1, observations.shape[1], observations.shape[2]), action, dtype=th.float32,
                     device=observations.device)
    return add_actions_to_obs(observations, action)


def match_action_with_obs_batch(observation_batch: th.Tensor, action_batch: list[int]):
    tensors = [th.full((1, 1, observation_batch.shape[2], observation_batch.shape[3]), action,
                       dtype=th.float32, device=observation_batch.device) for action in action_batch]
    actions = th.cat(tensors, dim=0)
    return add_actions_to_obs(observation_batch, actions, dim=1)


def resize_obs(observations: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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
    if isinstance(value, float) or isinstance(value, np.float32):
        scaled_v = np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1 + value * e)
        return np.array([scaled_v])
    return th.sign(value) * (th.sqrt(th.abs(value) + 1) - 1 + value * e)


def optuna_parameter_search(n_trials: int, init_net_path: str, storage: str, study_name: str):
    def objective(trial):
        num_mc_simulations = trial.suggest_int("num_mc_simulations", 100, 1200)
        num_self_play_games = trial.suggest_int("num_self_play_games", 100, 500)
        num_epochs = trial.suggest_int("num_epochs", 100, 500)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tau = trial.suggest_float("tau", 0.5, 1.5)
        arena_tau = trial.suggest_float("arena_tau", 0.01, 0.5)
        c = trial.suggest_float("c", 0.5, 5)
        c2 = trial.suggest_categorical("c2", [19652, 10_000, 0.01, 10, 0.1])
        K = trial.suggest_int("K", 1, 10)

        trial_args.num_simulations = num_mc_simulations
        trial_args.self_play_games = 5
        trial_args.epochs = num_epochs
        trial_args.lr = lr
        trial_args.tau = tau
        trial_args.c = c
        trial_args.c2 = c2
        trial_args.arena_tau = arena_tau
        trial_args.K = K
        trial_args.num_iters = 5

        game = Asteroids()
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        trial.net_action_size = int(game.get_num_actions())
        network = MuZeroNet.make_from_args(trial_args).to(device)
        network.load_state_dict(th.load(init_net_path))
        tree = MuZeroSearchTree(game.make_fresh_instance(), trial_args)
        net_player = NetPlayer(game.make_fresh_instance(),
                               **{"network": network, "monte_carlo_tree_search": tree})
        arena = MzArena(game.make_fresh_instance(), trial_args, device)
        trainer = Trainer.create(trial_args, game.make_fresh_instance(), network, tree, net_player,
                                 headless=True, arena_override=arena, checkpointer_verbose=False)
        trainer.train()
        win_freq = trainer.get_arena_win_frequencies_mean()
        trial.report(win_freq, trial.number)
        print(f"Trial {trial.number} finished with win freq {win_freq}.")
        del trainer
        del network
        del tree
        del net_player
        return win_freq

    from Game.asteroids import Asteroids
    from MuZero.MZ_Arena.arena import MzArena
    from MuZero.MZ_MCTS.mz_search_tree import MuZeroSearchTree
    from MuZero.Network.networks import MuZeroNet
    from AlphaZero.Network.trainer import Trainer
    trial_args = DotDict(SAMPLE_MZ_ARGS)
    trial_args.show_tqdm = False
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)

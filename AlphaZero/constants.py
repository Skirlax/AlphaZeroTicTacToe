# File for storing constants

board_size = 8

SAMPLE_ARGS = {
    "num_net_channels": 512,
    "num_net_in_channels": 1,
    "net_dropout": 0.3,
    "net_action_size": board_size ** 2,
    "num_simulations": 240,
    "self_play_games": 100,
    "num_iters": 1,
    "epochs": 100,
    "lr": 0.001,
    "max_buffer_size": 100_000,
    "num_pit_games": 40,
    "random_pit_freq": 3,
    "board_size": board_size,
    "batch_size": 256,
    "tau": 1,
    "arena_tau": 1e-2,
    "c": 1,
    "checkpoint_dir": None,
    "update_threshold": 0.6,
    "max_depth": float("inf"),
    "show_tqdm": True,
    "num_workers": 15,
    "num_to_win": 5,
    "log_epsilon": 1e-9,
    "zero_tau_after": 5
}

TRAINED_NET_ARGS = {
    "num_net_channels": 512,
    "num_net_in_channels": 1,
    "net_dropout": 0.3,
    "net_action_size": board_size ** 2,
    "num_simulations": 1317,
    "self_play_games": 145,
    "num_iters": 50,
    "epochs": 320,
    "lr": 0.0032485504583772953,
    "max_buffer_size": 100_000,
    "num_pit_games": 40,
    "random_pit_freq": 3,
    "board_size": board_size,
    "batch_size": 128,
    "tau": 1.0,
    "arena_tau": 0,  # 0.04139160592420218
    "c": 1.15,
    "checkpoint_dir": None,
    "update_threshold": 0.6,
    "max_depth": float("inf"),
    "show_tqdm": True,
    "num_workers": 5,
    "num_to_win": 3,
    "log_epsilon": 1.4165210108199043e-08,
}

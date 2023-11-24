import argparse
import json
import os
import subprocess

import optuna

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def find_biggest_divisor(n: int, t: int) -> int:
    # Might be used for spawning processes in the future.
    if t % n == 0:
        return n
    return find_biggest_divisor(n - 1, t)


def main():
    parser_ = argparse.ArgumentParser(
        description="Start optuna optimization jobs"
    )
    parser_.add_argument("n", help="Number of processes to start.")
    parser_.add_argument("t", help="Number of trials to run.")
    parser_.add_argument("-s", "--storage", help="The storage string to use.")
    parser_.add_argument("-n", "--study_name", help="The name of the optuna study to use.")

    args = parser_.parse_args()
    n = int(args.n)
    t = int(args.t)
    storage = args.storage
    if storage is None:
        storage = "mysql://alpha:584792@localhost/alpha_zero"
    study_name = args.study_name
    if study_name is None:
        study_name = "alpha_zero"

    init_net_path = "Checkpoints/NetVersions/h_search_network.pth"

    print(f"Starting optuna optimization. Using parameters: \n"
          f"Number of parallel processes: {n}\n"
          f"Number of trials: {t}\n"
          f"Storage: {storage}\n"
          f"Study name: {study_name}\n"
          f"Starting...")

    num_trials = t // n
    procs = []
    for i in range(n):
        procs.append(subprocess.Popen(
            ["python3.11", "main.py", "-s", storage, "-n", study_name, "-t", str(num_trials), "-i", init_net_path]))
    for proc in procs:
        proc.wait()

    study = optuna.load_study(study_name=study_name, storage=storage)
    with open("best_params.json", "w") as file:
        json.dump(study.best_params, file)


if __name__ == "__main__":
    main()

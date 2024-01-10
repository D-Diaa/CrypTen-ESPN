#!/usr/bin/env python3
import argparse
import logging
import os
import time
from datetime import timedelta

import torch
import yaml
from math import ceil

import crypten
from crypten import communicator as comm
from crypten.config import cfg
from examples.meters import AverageMeter
from examples.multiprocess_launcher import MultiProcessLauncher


def get_range(start, end, step):
    return torch.tensor([start + i * step for i in range(int(ceil(end - start) / step))])


def encrypt_data_tensor_with_src(input, device):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = crypten.comm.get().get_rank()
    # get world size
    world_size = crypten.comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0
    private_input = crypten.cryptensor(input.to(device), src=src_id).to(device)
    return private_input


def timeit(fn, size):
    comm.get().reset_communication_stats()
    start_time = time.monotonic()
    out = fn()
    end_time = time.monotonic()
    delta = timedelta(seconds=end_time - start_time)
    time_diff = delta.seconds + delta.microseconds / 1e6
    average_time_diff = time_diff / size
    return out, time_diff, average_time_diff, comm.get().get_communication_stats()


def relu_compare_run(n_points=32768, repeats=20, delay=0.0, device=torch.device("cpu")):
    cfg.load_config("configs/default16.yaml")
    cfg.mpc.real_shares = True
    cfg.mpc.real_triplets = True
    cfg.communicator.delay = delay

    coeffs = list(cfg.functions.relu_coeffs)
    std = 2
    method_names = ["crypten", "espn", "espn_old", "honeybadger", "honeybadger_old", "relu"]

    # Initialize time measurements dictionary
    time_measurements = {method_name: AverageMeter() for method_name in method_names}

    def compute_error(output, reference):
        err = torch.histc((reference - output).abs(), 10).cpu().numpy()
        err_min = torch.min((reference - output).abs()).item()
        err_max = torch.max((reference - output).abs()).item()
        return err, err_min, err_max

    def log_performance(method_name, timing, comm_metrics, err, err_min, err_max):
        # Format histogram data as integers
        err_formatted = ", ".join([f"{int(x)}" for x in err])

        logging.info(
            f"{method_name} time: {timing:.2f} ({time_measurements[method_name].value():.2f})| "
            f"rounds: {comm_metrics['rounds']} | bytes: {comm_metrics['bytes']} | "
            f"err: [{err_formatted}] \\in [{err_min:.5f}:{err_max:.5f}]\n"
        )

    for _ in range(repeats):
        # Data preparation
        x_plain = torch.normal(mean=0.0, std=torch.ones(n_points) * std).to(device)
        x_input = encrypt_data_tensor_with_src(x_plain, device)
        x_plain = x_input.get_plain_text()

        # Define computation methods
        def create_method_function(method_name):
            def method_function():
                if "relu" not in method_name:
                    cfg.functions.relu_method = "poly"
                    cfg.functions.poly_method = method_name
                else:
                    cfg.functions.relu_method = "exact"
                return x_input.relu().get_plain_text()

            return method_function

        o_n = sum(coeffs[i] * x_plain ** i for i in range(len(coeffs)))

        # Perform and log each computation method
        # Generate and perform each computation method
        methods = {method_name: create_method_function(method_name) for method_name in method_names}

        for method_name, method in methods.items():
            output, timing, avg, comm_metrics = timeit(method, x_input.size(0))
            time_measurements[method_name].add(timing, 1)
            err, err_min, err_max = compute_error(output, o_n)
            log_performance(method_name, timing, comm_metrics, err, err_min, err_max)

    return time_measurements



parser = argparse.ArgumentParser(description="CrypTen Poly Eval")
"""
    Arguments for Multiprocess Launcher
"""
parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)

parser.add_argument(
    "--world-size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)


def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    methods = ["crypten", "espn", "espn_old", "honeybadger", "honeybadger_old", "relu"]
    n_points = 32768
    delays = [0.000125, 0.025, 0.05, 0.1]
    devices = [torch.device("cpu"), torch.device("cuda:0")]
    aggregable_keys = ["run_time", "run_time_95conf_lower", "run_time_95conf_upper"]

    for device in devices:
        all_results = {conf: {
            key: [] for key in aggregable_keys
        } for conf in methods}
        os.makedirs(f"results/relus/{device}", exist_ok=True)
        for delay in delays:
            results = relu_compare_run(n_points=n_points, delay=delay, device=device)
            for i, conf in enumerate(methods):
                res = results[conf].mean_confidence_interval()
                for j, key in enumerate(aggregable_keys):
                    all_results[conf][key].append(res[j].item())
        if "RANK" in os.environ and os.environ["RANK"] == "0":
            for conf in methods:
                all_results[conf]['delays'] = delays
                with open(f"results/relus/{device}/{conf}_result.yaml", "w") as f:
                    yaml.dump(all_results[conf], f)


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)

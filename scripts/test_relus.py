#!/usr/bin/env python3
import argparse
import logging
import os
import time
from datetime import timedelta
from math import ceil

import torch
from crypten.config import cfg
import crypten
from examples.meters import AverageMeter
from examples.multiprocess_launcher import MultiProcessLauncher
from crypten import communicator as comm
import matplotlib.pyplot as plt


def get_range(start, end, step):
    return torch.tensor([start + i * step for i in range(int(ceil(end - start) / step))])


def encrypt_data_tensor_with_src(input):
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

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size(), device=input.device)
    private_input = crypten.cryptensor(input_upd, src=src_id, device=input.device)
    return private_input


def timeit(fn, size):
    comm.get().reset_communication_stats()
    start_time = time.monotonic()
    out = fn()
    end_time = time.monotonic()
    delta = timedelta(seconds=end_time - start_time)
    time_diff = delta.seconds * 1e3 + delta.microseconds / 1e3
    average_time_diff = time_diff / size
    return out, time_diff, average_time_diff, comm.get().get_communication_stats()


def relu_compare_run(n_points=32768, repeats=8, delay=0.0, device=torch.device("cpu")):
    cfg.load_config("configs/default10.yaml")
    cfg.communicator.delay = delay

    coeffs = list(cfg.functions.relu_coeffs)
    _range = 6
    time_c = AverageMeter()
    time_f = AverageMeter()
    time_h = AverageMeter()
    time_r = AverageMeter()
    for _ in range(repeats):
        x_plain = torch.rand(n_points, device=device) * 2 * _range - _range
        x_input = encrypt_data_tensor_with_src(x_plain)
        x_plain = x_input.get_plain_text()

        def crypten_poly():
            cfg.functions.relu_method = "poly"
            cfg.functions.poly_method = "crypten"
            ans = x_input.relu()
            return ans.get_plain_text()

        def florian():
            cfg.functions.relu_method = "poly"
            cfg.functions.poly_method = "florian"
            ans = x_input.relu()
            return ans.get_plain_text()

        def honeybadger():
            cfg.functions.relu_method = "poly"
            cfg.functions.poly_method = "honeybadger"
            ans = x_input.relu()
            return ans.get_plain_text()

        def relu():
            cfg.functions.relu_method = "exact"
            ans = x_input.relu()
            return ans.get_plain_text()

        o_n = sum(coeffs[i] * x_plain ** i for i in range(len(coeffs)))

        o_c, t_c, avg_c, comm_c = timeit(crypten_poly, x_input.size(0))
        time_c.add(t_c, x_input.size(0))
        o_f, t_f, avg_f, comm_f = timeit(florian, x_input.size(0))
        time_f.add(t_f, x_input.size(0))
        o_h, t_h, avg_h, comm_h = timeit(honeybadger, x_input.size(0))
        time_h.add(t_h, x_input.size(0))
        o_r, t_r, avg_r, comm_r = timeit(relu, x_input.size(0))
        time_r.add(t_r, x_input.size(0))

        err_f = (o_n - o_f).abs().sum().item() / n_points
        err_h = (o_n - o_h).abs().sum().item() / n_points
        err_c = (o_n - o_c).abs().sum().item() / n_points
        err_r = (o_n - o_r).abs().sum().item() / n_points
        to_print = [
            t_c, time_c.value(), comm_c['rounds'], comm_c['bytes'], err_c,
            t_f, time_f.value(), comm_f['rounds'], comm_f['bytes'], err_f,
            t_h, time_h.value(), comm_h['rounds'], comm_h['bytes'], err_h,
            t_r, time_r.value(), comm_r['rounds'], comm_r['bytes'], err_r,
        ]
        logging.info(
            "-----------------------------------------------------------------------------\n"
            "crypten time: {:.2f} ({:.2f})| rounds: {} | bytes: {} | err: {:.5f}\n"
            "florian time: {:.2f} ({:.2f})| rounds: {} | bytes: {} | err: {:.5f}\n"
            "honeybd time: {:.2f} ({:.2f})| rounds: {} | bytes: {} | err: {:.5f}\n"
            "exact_r time: {:.2f} ({:.2f})| rounds: {} | bytes: {} | err: {:.5f}\n"
            "-----------------------------------------------------------------------------"
            .format(*to_print)
        )
    return time_c.value(), time_f.value(), time_h.value(), time_r.value()


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
    # level = logging.DEBUG
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    relus = ['crypten', 'florian', 'honeybagder', 'relu']
    n_points = 32768
    delays = [0.0, 0.025, 0.05, 0.075, 0.1]
    devices = [torch.device("cpu"), torch.device("cuda:0")]
    for device in devices:
        relu_times = {k: [] for k in relus}
        relu_times['delays'] = delays
        relu_times['relus'] = relus
        for delay in delays:
            results = relu_compare_run(n_points=n_points, delay=delay, device=device)
            for ind in range(len(relus)):
                relu_times[relus[ind]].append(results[ind])
        for relu in relu_times['relus']:
            plt.plot(relu_times['delays'], relu_times[relu], label=relu)
        plt.xlabel("Artificial Delay (one way) in seconds")
        plt.ylabel("Runtime in seconds")
        plt.title(f"Relu evaluation for {n_points} points with 4 degree poly on {device}")
        plt.legend()
        plt.savefig(f"plots/relu_{device}_{os.environ['RANK']}.png")
        plt.close()


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

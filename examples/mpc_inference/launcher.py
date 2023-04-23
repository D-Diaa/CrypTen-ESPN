#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os

import torch

from examples.multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTen Multidataset Inference")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 1)",
)
parser.add_argument(
    "--model-location",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--model-type",
    default="resnet18",
    type=str,
    choices=["resnet18", "resnet32", "resnet50", "resnet110", "vgg16", "vgg16_bn", "minionn"],
    help="Model architecture",
)

parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "imagenet"],
    help="evaluation dataset",
)

parser.add_argument(
    "--config",
    default="configs/default.yaml",
    type=str,
    metavar="PATH",
    help="path to latest crypten config",
)

parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)
parser.add_argument(
    "--use-cuda",
    default=False,
    action="store_true",
    help="Run example on gpu",
)
parser.add_argument(
    "--resume",
    default=False,
    action="store_true",
    help="Resume training from latest checkpoint",
)
parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="skip plaintext evaluation",
)
parser.add_argument(
    "--evaluate-separately",
    default=False,
    action="store_true",
    help="evaluate private model separately",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from examples.mpc_inference.mpc_inference import run_mpc_model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_mpc_model(
        config = args.config,
        batch_size=args.batch_size,
        print_freq=args.print_freq,
        model_location=args.model_location,
        model_type=args.model_type,
        dataset=args.dataset,
        seed=args.seed,
        skip_plaintext=args.skip_plaintext,
        resume=args.resume,
        evaluate_separately=args.evaluate_separately,
        device=device
    )


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

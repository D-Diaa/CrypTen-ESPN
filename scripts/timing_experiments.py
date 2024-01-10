#!/usr/bin/env python3
import os

config_folder = "configs"
batch_size = 1
repeats = 20

# [LAN, MIDDLE, WAN]
# delays = ["0.000125", "0.025", "0.05"] #LOCAL WAN
delays = ["0"] #Actual WAN

datasets = ["cifar10", "cifar100", "imagenet"]

# models = {
#     "cifar10": ["resnet110", "minionn_bn", "vgg16_avg_bn", "resnet18"],
#     "cifar100": ["resnet32", "vgg16_avg_bn", "resnet18"],
#     "imagenet": ["resnet50"]
# }

models = {
    "cifar10": ["resnet110", "resnet18"],
    "cifar100": ["resnet32", "resnet18"],
    "imagenet": ["resnet50"]
}


configs = ["default12.yaml", "crypten12.yaml"] #CryptGPU, CryptenPoly
configs += ["honeybadger16.yaml", "espn16.yaml"] #Ours

device_commands = ["--use-cuda"]
#LOCAL WAN
# base_command = f"python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 " \
#ACTUAL WAN
base_command = f"python3 examples/mpc_inference/launcher.py  --world_size 2 " \ 
               f" --skip-plaintext " \
               f" --batch-size {batch_size} " \
               f" --n-batches {repeats}"

base_command += " --delays " + " ".join(delays)
cmds = []
for dataset in datasets:
    for model in models[dataset]:
        cmd = base_command + f" --dataset {dataset}" \
                             f" --model-type {model} "
        for device_cmd in device_commands:
            for config in configs:
                cmd += f" --config {config_folder}/{config} {device_cmd}"
                cmds.append(cmd)
os.system(";".join(cmds))

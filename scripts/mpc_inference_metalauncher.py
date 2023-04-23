import os

delays = ["0.0", "0.01", "0.05", "0.1"]
models = ["resnet18", "resnet32", "resnet50", "resnet110", "vgg16_bn", "minionn"]
datasets = ["cifar10", "cifar100"]
config_folder = "configs"
configs = ["default16.yaml", "crypten10.yaml", "florian10.yaml", "honeybadger10.yaml"]
device_commands = ["", "--use-cuda"]
base_command = "python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 --n-batches 3 --skip-plaintext"
base_command += " --delays " + " ".join(delays)

for dataset in datasets:
    for model in models:
        for device_cmd in device_commands:
            for config in configs:
                cmd = base_command + f" --dataset {dataset}" \
                                     f" --model-type {model}" \
                                     f" --config {config_folder}/{config}" \
                                     f" {device_cmd}"
                os.system(cmd)

import os

config_folder = "configs"
models_folder = "/home/a2diaa/trained_models/best_models"
batch_size = 32

delays = ["0.0"]
models = ["resnet18", "resnet32", "resnet50", "resnet110", "vgg16_bn", "minionn"]
datasets = ["cifar10", "cifar100"]

# configs = ["default16.yaml", "crypten10.yaml", "honeybadger10.yaml", "honeybadger8.yaml", "default10.yaml"]
configs = ["florian12.yaml", "default12.yaml", "crypten12.yaml", "honeybadger12.yaml"]

device_commands = ["--use-cuda"]

base_command = f"python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 " \
               f"--n-batches 1 " \
               f"--skip-plaintext " \
               f"--batch-size {batch_size}"

base_command += " --delays " + " ".join(delays)

for dataset in datasets:
    for model in models:
        cmd = base_command + f" --dataset {dataset}" \
                             f" --model-type {model} "
        for model_file in os.listdir(models_folder):
            if model_file.endswith(".pth") and model_file.startswith(f"{model}_{dataset}_"):
                cmd = base_command + f" --dataset {dataset}" \
                                     f" --model-type {model} " \
                                     f"--resume " \
                                     f"--model-location {models_folder}/{model_file}"
                break

        if cmd is not None:
            for device_cmd in device_commands:
                for config in configs:
                    cmd += f" --config {config_folder}/{config} {device_cmd}"
                    os.system(cmd)

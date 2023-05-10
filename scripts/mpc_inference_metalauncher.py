import os

config_folder = "configs"
models_folder = "/home/a2diaa/trained_models/best_models"
batch_size = 100

# delays = ["0.000125", "0.025", "0.05"]
# delays = ["0.000125", "0.01"]
delays = ["0.0"]

datasets = ["cifar100"]
#datasets = ["cifar100"]
# datasets = ["imagenet"]

models = {
    "cifar10": ["resnet110", "minionn_bn", "vgg16_bn", "resnet18"],
    "cifar100": ["resnet32", "vgg16_bn", "resnet18"],
    "imagenet": ["resnet50"]
}

models = {
    "cifar10": ["minionn_bn", "resnet18"],
    "cifar100": ["resnet32", "resnet18"],
    "imagenet":[]
}


configs = ["florian12.yaml", "default12.yaml", "crypten12.yaml", "honeybadger12.yaml"]
# configs = ["florian8.yaml", "default8.yaml", "crypten8.yaml", "honeybadger8.yaml"]
configs = ["honeybadger12.yaml"]

device_commands = ["--use-cuda"]

base_command = f"python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 " \
               f"--skip-plaintext " \
               f"--batch-size {batch_size}"

base_command += " --delays " + " ".join(delays)

for dataset in datasets:
    for model in models[dataset]:
        cmd = base_command + f" --dataset {dataset}" \
                             f" --model-type {model} "
        for model_folder in os.listdir(f"{models_folder}/{dataset}"):
            if model_folder.startswith(f"{model}"):
                model_file = f"{models_folder}/{dataset}/{model_folder}/run_1/best_model.pth"
                cmd = base_command + f" --dataset {dataset}" \
                                     f" --model-type {model} " \
                                     f"--resume " \
                                     f"--model-location {model_file}"
                break

        if cmd is not None:
            for device_cmd in device_commands:
                for config in configs:
                    cmd += f" --config {config_folder}/{config} {device_cmd}"
                    os.system(cmd)

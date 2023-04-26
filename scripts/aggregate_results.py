import os.path

import yaml
import matplotlib.pyplot as plt

# models = ["resnet18", "resnet32", "resnet50", "resnet110", "minionn", "vgg16_bn"]
# datasets = ["cifar10", "cifar100"]
models = ["resnet18"]
datasets = ["cifar10_batch4"]
config_folder = "configs"
configs = ["default16", "crypten10", "honeybadger10", "florian10"]
colors = ["r", "g", "k", "m"]
styles = ["-", "--"]
devices = ["cpu", "cuda:0"]

for dataset in datasets:
    for model in models:
        plot_label = f"{dataset}_{model}"
        # create a plot
        for s, device in enumerate(devices):
            for c, config in enumerate(configs):
                label = f"{config}_{device}"
                results_file = f"results/{dataset}/{model}_{device}/{config}_result.yaml"
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results = yaml.safe_load(f)
                    fmt_str = colors[c]+styles[s]
                    plt.plot(results['delays'], results['run_time'], fmt_str, label=label)
                else:
                    print(results_file + " is not found")
        # save plots and reset
        plt.title(f"Performance of {model} on {dataset}")
        plt.xlabel("Artificial delay in seconds")
        plt.ylabel("Total inference runtime in seconds")
        plt.legend()
        plt.savefig("plots/"+plot_label+".png")
        plt.close()

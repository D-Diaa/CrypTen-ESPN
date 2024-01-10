import os
import yaml
import pandas as pd

results_folder = 'results_timing_growl'
# Defining the datasets, models and methods to be included in the table
datasets = ['cifar10', 'cifar100', 'imagenet']
models = {
    'cifar10': ['minionn_bn_cuda', 'resnet110_cuda', 'resnet18_cuda', 'vgg16_avg_bn_cuda'],
    'cifar100': ['resnet32_cuda', 'resnet18_cuda', 'vgg16_avg_bn_cuda'],
    'imagenet': ['resnet50_cuda']
}
methods = ['honeybadger16_result', 'espn16_result', 'coinn_result', 'gforce_result',
           'opencheetah_result', 'default12_result']

renaming = {
    'honeybadger16_result': 'Honeybadger', 'espn16_result': 'ESPN', 'coinn_result': 'COINN', 'gforce_result': 'GForce',
    'opencheetah_result': 'CHEETAH', 'default12_result': "CryptGPU"
}


# Helper function to read YAML data
def read_yaml_data(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Creating a DataFrame to store the results
columns = ['Dataset', 'Model'] + [f"{method}_{env}" for method in methods for env in ['LAN', 'WAN']]
results_table = pd.DataFrame(columns=columns)

# Iterate through datasets, models, and methods to populate the DataFrame
for dataset in datasets:
    for model in os.listdir(os.path.join(results_folder, dataset)):
        row = {'Dataset': dataset, "Model":model}
        model_path = os.path.join(results_folder, dataset, model)

        for method in methods:
            method_path = os.path.join(model_path, f"{method}.yaml")

            if os.path.exists(method_path):
                data = read_yaml_data(method_path)
                lan_runtime = data['run_time'][0] if 'run_time' in data else None
                wan_runtime = data['run_time'][-1] if 'run_time' in data else None

                row[f"{method}_LAN"] = lan_runtime
                row[f"{method}_WAN"] = wan_runtime
            else:
                row[f"{method}_LAN"] = None
                row[f"{method}_WAN"] = None

        results_table = results_table._append(row, ignore_index=True)

results_table.to_csv("timing.csv", index=False)

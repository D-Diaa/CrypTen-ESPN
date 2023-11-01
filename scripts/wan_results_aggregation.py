import os
import yaml
import pandas as pd

results_folder = "results"
ROUND = 2

# List of machine paths (to average from)
machine_paths = [
    os.path.join(results_folder, machine, "../results") for machine in os.listdir(results_folder)
]

# List of configurations to consider
CONFIGURATIONS = ['espn12', 'honeybadger12', 'default12']

# Dictionary to rename configurations for the final table
RENAME_DICT = {
    'espn12': 'ESPN',
    'honeybadger12': 'HoneyBadger',
    'default12': 'CryptGPU'
}


def extract_data_from_machine(machine_folder_path):
    data = []
    for dataset in os.listdir(machine_folder_path):
        dataset_path = os.path.join(machine_folder_path, dataset)
        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            for config in CONFIGURATIONS:
                config_file = config + '_result.yaml'
                config_path = os.path.join(model_path, config_file)
                if os.path.exists(config_path):
                    with open(config_path, 'r') as file:
                        config_data = yaml.safe_load(file)
                        entry = {
                            "dataset": dataset,
                            "model": model.split("_")[0],
                            "config": RENAME_DICT[config],
                            "comm_time": round(config_data["comm_time"][0], ROUND),
                            "run_time_amortized": round(config_data["run_time_amortized"][0], ROUND),
                            "run_time_95conf_lower": round(config_data["run_time_95conf_lower"][0], ROUND),
                            "run_time_95conf_upper": round(config_data["run_time_95conf_upper"][0], ROUND)
                        }
                        data.append(entry)
    return data


# Extract, process, and format data as before
combined_data = []
for path in machine_paths:
    combined_data += extract_data_from_machine(path)

# Process data
df = pd.DataFrame(combined_data)
aggregated_data = df.groupby(['dataset', 'model', 'config']).agg({
    'comm_time': 'mean',
    'run_time_amortized': 'mean',
    'run_time_95conf_lower': 'mean',
    'run_time_95conf_upper': 'mean'
}).reset_index()
aggregated_data['confidence_interval'] = aggregated_data['run_time_95conf_upper'] - aggregated_data[
    'run_time_95conf_lower']
table = aggregated_data.pivot_table(index=['dataset', 'model'],
                                    columns='config',
                                    values=['comm_time', 'run_time_amortized', 'confidence_interval'])
table = table.reorder_levels([1, 0], axis=1)
table.sort_index(axis=1, level=[0, 1], inplace=True)

for config in CONFIGURATIONS:
    renamed_config = RENAME_DICT[config]
    table[(renamed_config, 'comm_time')] = "$" + round(table[(renamed_config, 'comm_time')], ROUND).astype(str) + "$"
    table[(renamed_config, 'run_time')] = ("$" + round(table[(renamed_config, 'run_time_amortized')], ROUND).astype(str)
                                           + " \\pm " + round(table[(renamed_config, 'confidence_interval')] / 2, ROUND).astype(str) + "$")

# Split into comm table and run_time table
comm_table = table[[col for col in table.columns if 'comm_time' in col]].copy()
run_time_table = table[[col for col in table.columns if 'run_time' in col]].copy()


# Convert tables to LaTeX format with styling, captions, and centering
comm_latex = comm_table.to_latex(escape=False, multicolumn_format='c', bold_rows=True,
                                 caption="Communication Time Table", label="tab:comm_time").replace("_time ", " time ")
run_time_latex = run_time_table.to_latex(escape=False, multicolumn_format='c', bold_rows=True,
                                         caption="Run Time Table", label="tab:run_time").replace("_time ", " time ")

print(comm_latex)
print(run_time_latex)

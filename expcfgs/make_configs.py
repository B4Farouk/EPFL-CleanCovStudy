import json
import os

def replace_and_save_config(estimator_names, cleaner_names, base_config):
    for estimator_name in estimator_names:
        for cleaner_name in cleaner_names:
            # Create a copy of the base configuration
            modified_config = base_config.copy()

            # Replace estimator and cleaner names
            modified_config["strategy"]["init_config"]["cov_cfg"]["estimator"]["name"] = estimator_name
            modified_config["strategy"]["init_config"]["cov_cfg"]["cleaner"]["name"] = cleaner_name if cleaner_name else None

            # Construct filename based on estimator and cleaner names
            filename = f"{estimator_name}_{cleaner_name}_MVO.json" if cleaner_name else f"{estimator_name}_None_MVO.json"

            # Specify directory where modified configs will be saved
            save_directory = "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/configs/johnny/"  # Replace with your desired directory

            # Save modified config as JSON
            with open(os.path.join(save_directory, filename), "w") as file:
                json.dump(modified_config, file, indent=4)

# Your provided configuration
base_configuration = {
    "strategy": {
        "name": "mean-variance",
        "init_config": {
            "cov_cfg":{
                "estimator": {
                    "name": "natural",
                    "params": {}
                },
                "cleaner": {
                    "name": "optimal_shrinkage",
                    "params": {}
                }
            },
            "solver_cfg":{
                "weight_bounds": [-0.2, 1.2],
                "verbose": False
            }
        }
    },
    "training_env":{
        "window_days_advancement": 7,
        "window_days_length": 90,
        "window_min_samples": 240,
        "test_size": 0.1,
        "window_max_nan_pct": 0.2,
        "save_dir": "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/results/",
        "use_multiproc": False,
        "multiproc_config": {"processes": 2}
    },
    "data_loader": {
        "init_config": {
            "data_path": "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/data/processed/v01"
        },
        "mode": "as_merged_dataframe"
    },
    "logging": {
        "path": "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/results/logs/debug_logs.log",
        "level": "DEBUG"
    },
    "use_dask": True
}

# Provided lists of estimator and cleaner names
estimator_list = ["natural", "ewm1"]
cleaner_list = [None, "clipping", "linear_shrinkage", "oas", "optimal_shrinkage", "bahc"]

# Call function to replace names and save configurations
replace_and_save_config(estimator_list, cleaner_list, base_configuration)

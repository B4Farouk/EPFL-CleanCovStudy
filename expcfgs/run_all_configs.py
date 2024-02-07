import os
import subprocess

def replace_config_names(config_directory, base_script_path):
    # Get a list of all JSON files in the specified directory
    config_files = [file for file in os.listdir(config_directory) if file.endswith(".json")]

    # Read the content of the base script
    with open(base_script_path, "r") as file:
        base_script = file.read()

    # Replace occurrences of debug_config_johnny.json in the script with config filenames and execute
    for config_filename in config_files:
        # Replace occurrences of debug_config_johnny.json with config filenames
        modified_script = base_script.replace("debug_config_johnny.json", config_filename)

        # Run the modified script
        subprocess.run(["zsh", "-c", modified_script])  # Execute the modified script

# Path to the directory containing the JSON config files
config_directory_path = "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/configs/johnny/"  # Replace with your config directory path

# Path to the target script that references debug_config_johnny.json for example
target_script_path = "/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/slurm/debug_local_johnny.sh"  # Replace with your target script path

# Call function to replace debug_config_johnny.json with generated config names and execute the modified script
replace_config_names(config_directory_path, target_script_path)

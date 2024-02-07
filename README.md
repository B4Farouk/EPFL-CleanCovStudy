## Data
The data used is the provided s&p100 tickers from 2005-2008 (by professor Damien Challet). Make sure to unzip it then place it in the `data/raw` folder at the root level of your project directory.

## Environment
Use the .requirements folder to create a virtual environment and run the code in it.

## Running the code
Ideally, run the code using the bash scripts available in the `slurm/` folder. Look at the provided example scripts and change them accordingly to suit your local machine. You will basically link a config file with every run. 

If you'd like to run a whole bunch of configs sequentially without having the rerun the script manually every time, I invite you to look at `configs/run_all_configs.py`. Make sure to change `config_directory_path` and `target_script_path` in the code and you should be all set.


Note that all config files we've used are available in `configs/johnny/`. Do feel free to make your own (for example by changing the limit allowed to short/long stocks..) but make sure to update the paths accordingly in the run scripts.
N.B: in all config files make sure you also change all paths accordingly.# EPFL-CleanCovStudy
# EPFL-CleanCovStudy

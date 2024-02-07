PROJECT_PATH="/Users/tchanee/Documents/EPFL/MA3/Financial Big Data/project/"
CONFIG_FILENAME="debug_config_johnny.json"
# source "$PROJECT_PATH"".venv/bin/activate"

conda init zsh
. /Users/tchanee/.zshrc
conda activate /opt/homebrew/Caskroom/miniconda/base/envs/fbd


python "$PROJECT_PATH""src/run.py" --config_path "$PROJECT_PATH/configs/johnny/$CONFIG_FILENAME"
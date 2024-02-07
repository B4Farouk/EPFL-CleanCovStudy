USERNAME=$1
CONFIG_FILENAME=$2
PROJECT_PATH="/home/$USERNAME/fbd/"
source "$PROJECT_PATH""venv/bin/activate"
python "$PROJECT_PATH""src/run.py" --config_path "$PROJECT_PATH/configs/$CONFIG_FILENAME"
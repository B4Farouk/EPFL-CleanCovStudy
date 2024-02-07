PROJECT_PATH="/home/farouk/Bureau/MA_3/fbd/repo/"
CONFIG_FILENAME=$1
source "$PROJECT_PATH"".venv/bin/activate"
python "$PROJECT_PATH""src/run.py" --config_path "$PROJECT_PATH/configs/$CONFIG_FILENAME"
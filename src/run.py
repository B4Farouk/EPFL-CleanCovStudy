import logging
import argparse
import json
import os
from typing import Dict, Any

import main
from utils import get_or_create_logger

def __log_config_info(
    logger: logging.Logger, config: Dict[str, Any], depth_prefix: str="\t"):
    def rec_config_repr(config: Dict[str, Any], depth_prefix: str):
        config_repr = ""
        for key, value in config.items():
            if isinstance(value, dict):
                continuation = rec_config_repr(config=value, depth_prefix=depth_prefix+"\t")
                continuation = '\n'.join([f"{depth_prefix} {key} ->", continuation])
            else:
                continuation = f"{depth_prefix} {key} -> {value}"
            config_repr = '\n'.join([config_repr, continuation])
        return config_repr
    
    logger.info("Loaded configuration.")
    
    if logger.level <= logging.DEBUG:
        config_repr = rec_config_repr(config=config, depth_prefix=depth_prefix)
        config_repr = '\n'.join(["Configuration:", config_repr])
        logger.debug(config_repr)

def __parser() -> argparse.ArgumentParser:    
    parser = argparse.ArgumentParser(description="FBD Project Script")
    
    # parameters
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    
    return parser

if __name__ == "__main__":
    args = __parser().parse_args()
    
    # Load config
    with open(args.config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # Run main
    logging_config = config.get("logging", None)
    logging_level  = logging_config.get("level", logging.INFO)
    logging_path_or_io_wrapper  = logging_config.get("path", None)
    if logging_path_or_io_wrapper is not None:
        os.makedirs(os.path.dirname(logging_path_or_io_wrapper), exist_ok=True)
    
    root_logger = get_or_create_logger(
        name=None, # use the root logger
        path_or_io_wrapper=logging_path_or_io_wrapper,
        level=logging_level
        )
    root_logger.info(f"Root logging level {root_logger.level}")
    
    __log_config_info(logger=root_logger, config=config)
    main.main(config=config)
    logging.shutdown()
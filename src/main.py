import logging
import pandas
import time
from typing import Dict, Any

from loaders.data_loaders import DataLoader
from strategy.factory import get_strategy
from local_dask import LocalDaskManager
from utils import get_or_create_logger
from training_env import TrainingEnv

############################
# Auxilary functions
############################
            
def __log_data_info(logger: logging.Logger, data: pandas.DataFrame):
    logger.info("Loaded data.")
    
    if logger.level <= logging.DEBUG:
        data_repr = f"""Data summary:
        \t Number of Rows   : {len(data)}
        \t Number of Columns: {len(data.columns)}
        \t Columns          : {list(data.columns)}
        """     
        logger.debug(data_repr)
        
############################
# Main function
############################

def main(config: Dict[str, Any]):
    # Create logger. Will use the same level as the root logger in run.py.
    logger = get_or_create_logger(
        name=__name__, 
        path_or_io_wrapper=config.get("logging_path", None)
    )
    
    # Create Dask client and cluster if needed
    use_dask = config.get("use_dask", False)
    cluster, client = None, None
    if use_dask:
        cluster = LocalDaskManager.get_or_create_cluster()
        client  = LocalDaskManager.get_or_create_client()
        logger.debug("Dask local cluster and client created.")
    
    # Start timer
    start_time = time.time()
    
    # Load data
    data_loader_config = config["data_loader"]
    data_loader = DataLoader(config=data_loader_config["init_config"], use_dask=use_dask)
    data = None
    match data_loader_config["mode"]:
        case "as_merged_dataframe":
            data, _ = data_loader.as_merged_dataframe(keep_cols="ret") # using returns, not log returns
        case _:
            raise NotImplementedError(f"Data loader mode {data_loader_config['mode']} not implemented")
    __log_data_info(logger=logger, data=data)
    
    # Load strategy
    strategy_config = config["strategy"]
    strategy_name = strategy_config["name"]
    strategy_init_config = strategy_config["init_config"]
    strategy = get_strategy(name=strategy_name, cfg=strategy_init_config)
    strategy_id_initials = strategy.get_strategy_id()[:5]
    logger.info("Loaded strategy %s (ID=%s).", strategy_name, strategy_id_initials)
    logger.info("Strategy covariance-related configuration: \n %s", strategy_init_config)
    
    # Load training environment
    training_env_config = config["training_env"]
    training_env = TrainingEnv(**training_env_config)
    logger.debug("Loaded training environment with configuration: \n %s", training_env_config)

    # Run the strategy
    logger.info("Started running the strategy in the training environment...")
    training_env.run(
        data=data,
        strategy=strategy)
    logger.info("Finished running the strategy in the training environment.")
    
    if config["use_dask"]:
        client.close()
        cluster.close()
        logger.debug("Closed dask client and cluster.")
    
    # compute the running time in (hours, minutes, seconds)
    end_time = time.time()
    running_time_s = end_time - start_time
    running_time_h, running_time_m = divmod(running_time_s, 3600)
    running_time_m, running_time_s = divmod(running_time_m, 60)
    total_running_time_msg = "Total running time: {:02d}:{:02d}:{:02d}".format(
        int(running_time_h), int(running_time_m), int(running_time_s))
    logger.info(total_running_time_msg)
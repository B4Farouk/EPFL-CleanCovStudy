import pathlib
import multiprocessing
import datetime as dt
import logging
import os
import math
import jsonlines as jsonl
from typing import Optional, Dict, Any

import pandas

from utils import get_or_create_logger
from strategy.base import Strategy
from evaluation.numeric import NumericEvaluator, NumericResults
from pypfopt.exceptions import OptimizationError

class TrainingEnv:
    def __init__(
        self,
        window_days_length: int,
        window_days_advancement: int,
        window_min_samples: int,
        save_dir: str | pathlib.Path,
        test_size: float =0.1,
        window_max_nan_pct: float =0.2,
        use_multiproc: bool =False,
        multiproc_config: Dict[str, Any] =None,
        logger: Optional[logging.Logger] =None):
        
        assert window_days_length > 0, "n_days_in_window must be positive"
        assert 0 < test_size < 1, "Test size must be between 0 and 1"
        assert 0 < window_max_nan_pct < 1, "window_max_nan_pct must be between 0 and 1"
        assert isinstance(save_dir, str) or isinstance(save_dir, pathlib.Path), "save_dir must be a string or a pathlib.Path"
        
        self.window_days_length = window_days_length
        self.window_days_advancement = window_days_advancement
        self.window_min_samples = window_min_samples
        self.test_size = test_size
        self.window_max_nan_pct = window_max_nan_pct
        self.save_dir = save_dir
        
        self.logger = (
            logger if logger is not None
            else get_or_create_logger(
                name=self.__class__.__name__,
                path_or_io_wrapper=os.path.join(save_dir, "logs", "training_env.log"),
                level=logging.INFO)
        )
        
        self.process_pool = None
        if use_multiproc:
            if multiproc_config is None:
                multiproc_config = {
                    "processes": max(multiprocessing.cpu_count() - 2, 1)
                }
            self.process_pool = multiprocessing.Pool(**multiproc_config)
            self.logger.debug("Created process pool.")
        else:
            self.logger.debug("No multiprocessing.")
        
    def run(self, data: pandas.DataFrame, strategy: Strategy):
        assert isinstance(data, pandas.DataFrame), "data must be a pandas.DataFrame instance"
        assert isinstance(strategy, Strategy), "strategy must be a Strategy instance"
        
        strategy_id = strategy.get_strategy_id()
        strategy_id_initials = strategy_id[:5]
        weights_savepath = os.path.join(
            self.save_dir, "weights", f"strategy_{strategy_id_initials}_weights.jsonl")
        results_savepath = os.path.join(
            self.save_dir, "numeric", f"strategy_{strategy_id_initials}_results.jsonl")
        cov_mat_savepath = os.path.join(
            self.save_dir, "cov", f"strategy_{strategy_id_initials}_cov.jsonl")
        
        assert not os.path.exists(weights_savepath), f"{weights_savepath} already exists. Training Environment refrained from altering it."
        assert not os.path.exists(results_savepath), f"{results_savepath} already exists. Training Environment refrained from altering it."
        
        num_evaluator = NumericEvaluator(strategy_id=strategy_id)
        
        start_date = data.index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        last_date  = data.index[-1].replace(hour=23, minute=59, second=59, microsecond=59)
        
        self.logger.info("First date is %s", start_date.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Last date is %s", last_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        window_start_date = start_date
        window_size_dt = dt.timedelta(days=self.window_days_length)
        window_end_date = window_start_date + window_size_dt
        
        self.logger.info("Initial window start date is %s", window_start_date.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Initial window end date date is %s", window_end_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        window_advancement_dt = dt.timedelta(days=self.window_days_advancement)
        one_day = dt.timedelta(days=1)
        iteration = 1
        window_n  = 1
        while window_end_date <= last_date:
            # Get the window data
            data_after_window_start_date = data.loc[data.index >= window_start_date]
            window_data = data_after_window_start_date.loc[data_after_window_start_date.index < window_end_date]
            
            # If the window is too small, extend it by one day at a time until it is big enough
            n_extensions = 0
            window_length = len(window_data)
            while window_length < self.window_min_samples:
                if window_end_date > last_date:
                    self.logger.info("Window %s has reached the last date without satisfying the minimum number of samples."
                                      "It has length %s and the minimum required is %s.",
                                      window_n, window_length, self.window_min_samples)
                    # If we are at the last date, we cannot extend the window anymore.
                    # Here "return" is needed (not "break") to avoid operating on a window that is too small
                    return
                window_end_date += one_day
                window_data = data_after_window_start_date.loc[data.index < window_end_date]
                window_length = len(window_data)
                n_extensions += 1
                
            if n_extensions > 0:
                self.logger.info("Window %s has been extended %s times in total for having less than the minimum number of samples %s",
                                  window_n, n_extensions, self.window_min_samples)
            
            solved = self.__train_eval_on_window(
                    window_data=window_data,
                    strategy=strategy,
                    num_evaluator=num_evaluator,
                    weights_savepath=weights_savepath,
                    cov_mat_savepath=cov_mat_savepath,
                    results_savepath=results_savepath,
                    iteration=iteration)
                
            iteration += 1
            
            if solved:
                # Advance the window
                self.logger.debug("Window %s has been solved. It has length %s",
                                  window_n, window_length)
                window_n += 1
                window_end_date += window_advancement_dt
                window_start_date = window_end_date - window_size_dt # window_start_date so that we have a constant size window
            else:
                # If the strategy is not solvable, we advance the window by one day
                # so that hopefully the strategy will be solvable in the next window
                self.logger.debug("Window %s has not been solved. It will be extended. Currently it has length %s",
                                  window_n, window_data)
                window_end_date += one_day
                window_start_date = window_end_date - window_size_dt # window_start_date so that we have a constant size window
            
            self.logger.debug("End of iteration %s", iteration)
        
        self.logger.info("Total number of iterations is %s", iteration)    
        self.logger.info("Total number of windows is %s", window_n)
        
        if self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()
            self.logger.debug("Joined process pool.")
            
        self.logger.info("Training environment has finished.")
        
    @staticmethod
    def __save_cov_matrix(
        cov_dict: dict,
        start_date: dt.datetime | pandas.Timestamp,
        test_start_date: dt.datetime | pandas.Timestamp,
        end_date: dt.datetime | pandas.Timestamp,
        path: str, write_mode: str ='a'):
        
        assert write_mode in ['a', 'w'], "write_mode must be either 'a' or 'w'"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "start_date": start_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "test_start_date": test_start_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "assets": cov_dict["assets"],
            "cov": cov_dict["cov"]
        }
        with jsonl.open(path, mode=write_mode) as writer:
            writer.write(data)
 
    @staticmethod
    def __save_weights(
        weights_dict: dict,
        start_date: dt.datetime | pandas.Timestamp,
        test_start_date: dt.datetime | pandas.Timestamp,
        end_date: dt.datetime | pandas.Timestamp,
        path: str, write_mode: str ='a'):
        
        assert write_mode in ['a', 'w'], "write_mode must be either 'a' or 'w'"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "start_date": start_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "test_start_date": test_start_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime(format="%Y-%m-%d %H:%M:%S"),
            "assets": list(weights_dict.keys()),
            "weights": list(weights_dict.values())
        }
        with jsonl.open(path, mode=write_mode) as writer:
            writer.write(data)
    
    @staticmethod
    def __save_result(result: NumericResults, path: str, write_mode: str ='a'):
        assert write_mode in ['a', 'w'], "write_mode must be either 'a' or 'w'"
        result.write(path=path, mode=write_mode)
        
    def __clean_window_data(self, window_data: pandas.DataFrame) -> pandas.DataFrame:
        # Remove columns with too many NaNs
        n_samples = len(window_data)
        max_nan = math.ceil(self.window_max_nan_pct * n_samples)
        threshold = n_samples - max_nan
        window_data = window_data.dropna(axis=1, thresh=threshold)
        
        # Backward fill NaNs
        window_data = window_data.bfill()
        
        # Sanity checks
        assert len(window_data.columns) > 0, "No asset left after cleaning"
        
        return window_data

    def __train_eval_on_window(
        self,
        window_data: pandas.DataFrame,
        strategy: Strategy,
        num_evaluator: NumericEvaluator,
        weights_savepath: str,
        cov_mat_savepath: str,
        results_savepath: str,
        iteration: int) -> bool:
        # Sanity checks
        assert len(window_data.columns) > 0, "window_data must have at least one column"
        
        # Clean the window data
        all_assets = list(window_data.columns)
        window_data = self.__clean_window_data(window_data)
        retained_assets = list(window_data.columns)
        
        # Set extra parameters for the covariance cleaner
        if strategy.cov_estimator_name == "ewm1":
            strategy.set_cov_estimator_params(params={
                "target_assets_mask": [asset in retained_assets for asset in all_assets]
        }, overwrite=True)
        
        # Split data into train and test
        test_length = int(len(window_data) * self.test_size)
        
        train_data = window_data.iloc[:-test_length]
        test_data = window_data.iloc[-test_length:]
        
        window_start_date = train_data.index[0]
        window_test_start_date = test_data.index[0]
        window_end_date   = test_data.index[-1]
        
        # Fit the strategy
        weights, cov = None, None
        try:
            weights, cov = strategy.optimize(data=train_data)
        except OptimizationError as err:
            self.logger.debug("Error: %s occured at iteration %s.",
                                  err.__class__.__name__, iteration)
        
        solved = weights is not None
        if not solved:
            self.logger.info("Strategy is not solvable between %s and %s at iteration %s",
                             window_start_date.strftime("%Y-%m-%d %H:%M:%S"),
                             window_end_date.strftime("%Y-%m-%d %H:%M:%S"),
                             iteration)
            return False
        
        assets = list(train_data.columns)
        weights_dict = dict(zip(assets, weights))
        cov_dict = {"assets": assets, "cov": cov.tolist()}
        
        # Save the weights and covariance matrix
        weight_saving_params = {
            "weights_dict": weights_dict,
            "start_date": window_start_date,
            "test_start_date": window_test_start_date,
            "end_date": window_end_date,
            "path": weights_savepath
        }
        cov_mat_saving_params = {
            "cov_dict": cov_dict,
            "start_date": window_start_date,
            "test_start_date": window_test_start_date,
            "end_date": window_end_date,
            "path": cov_mat_savepath
        }
        if self.process_pool is not None:
            self.process_pool.apply_async(TrainingEnv.__save_weights, kwds=weight_saving_params)
            self.process_pool.apply_async(TrainingEnv.__save_cov_matrix, kwds=cov_mat_saving_params)
            self.logger.debug("Submitted the job: weight saving at iteration %s", iteration)
        else:
            TrainingEnv.__save_weights(**weight_saving_params)
            TrainingEnv.__save_cov_matrix(**cov_mat_saving_params)
            self.logger.debug("Saved the weights at iteration %s", iteration)
            
        # Evaluate the strategy
        result = num_evaluator.evaluate(
            test_data=test_data,
            weights_dict=weights_dict,
            cov_dict=cov_dict,
            start_date=window_start_date,
            test_start_date=window_test_start_date,
            end_date=window_end_date)
        
        # Save the result
        save_result_params = {
            "result": result,
            "path": results_savepath
        }
        if self.process_pool is not None:
            self.process_pool.apply_async(TrainingEnv.__save_result, kwds=save_result_params)
            self.logger.debug("Submitted the job: result saving at iteration %s", iteration)
        else:
            TrainingEnv.__save_result(**save_result_params)
            self.logger.debug("Saved result at iteration %s", iteration)
        
        return True
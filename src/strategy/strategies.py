from typing import Dict, Any

import pypfopt
import pandas
import torch
import numpy
import numpy as np

from strategy.base import Strategy
from covariance.estimation import cov_estimator_is_regressive
from utils import to_torch_tensor

class MeanVarianceStrategy(Strategy):    
    def __init__(self,
                 cov_cfg: Dict[str, Any],
                 solver_cfg: Dict[str, Any]):
        super().__init__(cov_cfg)
        
        weight_bounds = list(solver_cfg["weight_bounds"])
        assert len(weight_bounds) == 2 and sum(weight_bounds) == 1 
        self.weight_bounds = tuple(weight_bounds)
        
        self.verbose = bool(solver_cfg["verbose"])
    
    def optimize(self,
                 data: pandas.DataFrame,
                 column_assets: bool =True) -> numpy.ndarray:
        data = to_torch_tensor(data.to_numpy())
        data = data if column_assets else data.T
        
        # compute covariance matrix
        cleaning_result = self.cov_cleaner(X=data, column_assets=True, **self.cov_cleaner_params)
        cov, cov_prev = None, None
        if cov_estimator_is_regressive(self.cov_estimator_name):
            cov, cov_prev = cleaning_result
            self.cov_cleaner_params["estimator"]["params"]["cov_prev"] = cov_prev
        else:
            cov = cleaning_result
        assert cov is not None
        
        # compute mean vector
        mean = data.mean(axis=0)
        
        if isinstance(mean, torch.Tensor):
            mean = mean.numpy()
        else:
            assert isinstance(mean, np.ndarray)
            
        if isinstance(cov, torch.Tensor):
            cov  = cov.numpy()
        else:
            assert isinstance(cov, np.ndarray)
        
        efficient_frontier = pypfopt.EfficientFrontier(
            expected_returns=mean,
            cov_matrix=cov,
            weight_bounds=self.weight_bounds,
            verbose=self.verbose)
        
        weights = efficient_frontier.max_sharpe(risk_free_rate=0.0)
        weights = list(weights.values())
        return weights, cov
    
    def set_cov_estimator_params(self, params: Dict[str, Any], overwrite: bool =False):
        for key, value in params.items():
            if (key in self.cov_cleaner_params["estimator"]["params"]) and not overwrite:
                raise ValueError(f"Parameter {key} already exists in the covariance estimator params.")
            self.cov_cleaner_params["estimator"]["params"][key] = value

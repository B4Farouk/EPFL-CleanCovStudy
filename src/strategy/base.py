from abc import ABC, abstractmethod

from typing import Dict, Any, Tuple

import pandas
import numpy

from covariance.estimation import get_covariance_estimator
from covariance.cleaning import get_covariance_cleaner

class Strategy(ABC):
    @staticmethod
    def create_strategy_id(
        class_name,
        cov_estimator_name: str,
        cov_cleaner_name: str,
        cov_estimator_params: Dict[str, Any],
        cov_cleaner_params: Dict[str, Any]
        ) -> str:
        import xxhash
        hash = xxhash.xxh3_128()
        
        str_attrs = [
            class_name,
            cov_estimator_name, str(cov_estimator_params),
            cov_cleaner_name, str(cov_cleaner_params)
        ]
        for attr in str_attrs:
            if attr: # since attribute can be None
                hash.update(attr.encode("utf-8"))
            
        hash = hash.hexdigest()
        return hash.lower()
    
    def __init__(self, cov_cfg: Dict[str, Any]):
        # Covariance cleaner, including covariance estimator and params
        self.cov_estimator_name = cov_cfg["estimator"]["name"]
        self.cov_cleaner_name = cov_cfg["cleaner"]["name"]
        
        self.cov_cleaner_params = cov_cfg["cleaner"].get("params", {})
        self.cov_cleaner_params["estimator"] = cov_cfg["estimator"] # add estimator params to cleaner params
        self.cov_cleaner = get_covariance_cleaner(cleaner=self.cov_cleaner_name)
        
        print(self.cov_cleaner_params)
        
        # Natural covariance estimator can be useful for regressive covariance estimators
        self.cov_estimator_natural = get_covariance_estimator(estimator="natural")
        
        self.strategy_id = Strategy.create_strategy_id(
            self.__class__.__name__,
            cov_estimator_name=self.cov_cleaner_params["estimator"]["name"],
            cov_cleaner_name=self.cov_cleaner_name,
            cov_estimator_params=self.cov_cleaner_params["estimator"]["params"],
            cov_cleaner_params=self.cov_cleaner_params
        )
        
        self.cov_cleaner_params["estimator"]["params"]["cov_prev"] = None
              
    def get_strategy_id(self) -> str:
        return self.strategy_id
    
    @abstractmethod
    def optimize(self,
                 data: pandas.DataFrame,
                 column_assets: bool =True) -> Tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError(f"{self.__class__.__name__}::optimize is not implemented")
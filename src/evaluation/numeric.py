from __future__ import annotations

import os
import pathlib
import jsonlines as jsonl
import datetime as dt
import pandas
import pandas as pd
import torch
from operator import itemgetter

from dataclasses import dataclass
from typing import List, Dict

from utils import to_torch_tensor


@dataclass
class NumericResults:
    strategy_id: str
    start_date: dt.datetime | pd.Timestamp
    test_start_date: dt.datetime | pd.Timestamp
    end_date: dt.datetime | pd.Timestamp
    assets: List[str]
    return_per_asset: List[float]
    return_: float
    std_per_asset: List[float]
    std: float
    
    def __to_dict(self) -> dict:
        start_date = self.start_date.strftime(format="%Y-%m-%d %H:%M:%S")
        end_date = self.end_date.strftime(format="%Y-%m-%d %H:%M:%S")
        test_start_date = self.test_start_date.strftime(format="%Y-%m-%d %H:%M:%S")
        
        return_per_asset = [float(x) for x in self.return_per_asset]
        return_ = float(self.return_)
        
        std_per_asset = [float(x) for x in self.std_per_asset]
        std = float(self.std)
        
        return {
            "strategy_id": self.strategy_id,
            "start_date": start_date,
            "test_start_date": test_start_date,
            "end_date": end_date,
            "assets": self.assets,
            "return_per_asset": return_per_asset,
            "return": return_,
            "std_per_asset": std_per_asset,
            "std": std
            }
    
    @classmethod
    def __from_dict(cls, data: dict) -> NumericResults:
        return {
            "strategy_id": str(data["strategy_id"]),
            "start_date": dt.datetime.strptime(data["start_date"], format="%Y-%m-%d %H:%M:%S"),
            "test_start_date": dt.datetime.strptime(data["test_start_date"], format="%Y-%m-%d %H:%M:%S"),
            "end_date": dt.datetime.strptime(data["end_date"], format="%Y-%m-%d %H:%M:%S"),
            "assets": list(data["assets"]),
            "return_per_asset": [float(x) for x in data["return_per_asset"]],
            "return_": float(data["return"]),
            "std_per_asset": [float(x) for x in data["std_per_asset"]],
            "std": float(data["std"])
        }
    
    def write(self, path: str | pathlib.Path, mode: str ='a'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with jsonl.open(path, mode=mode) as writer:
            writer.write(self.__to_dict())
        
    @classmethod    
    def read_all(cls, path: str | pathlib.Path) -> List[NumericResults]:
        results = None
        with jsonl.open(path, mode='r') as reader:
            results = [NumericResults.__from_dict(line) for line in reader]
            
        # sort them by start date
        results = [(result.start_date, result) for result in results]
        results = sorted(results, key=itemgetter(0))
        results = [result for _, result in results]
        return results


class NumericEvaluator:
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
    
    @staticmethod
    def __get_returns(
        test_data: torch.Tensor,
        weights: torch.Tensor) -> tuple[float, List[float]]:
        
        weighted_asset_returns = test_data * weights.view(1, -1)
        
        aggregate_return_per_step = weighted_asset_returns.sum(axis=1).ravel()
        aggregate_return = (1 + aggregate_return_per_step).prod() - 1
        aggregate_return = aggregate_return.item()
        
        aggregate_return_per_asset = (1 + weighted_asset_returns).prod(axis=0) - 1
        aggregate_return_per_asset = aggregate_return_per_asset.ravel().numpy().tolist()
        
        return aggregate_return, aggregate_return_per_asset
    
    @staticmethod
    def __get_standard_devs(
        weights: torch.Tensor,
        cov: torch.Tensor) -> tuple[float, List[float]]:
        weights = weights.reshape(-1, 1)
        std = torch.sqrt(weights.T @ cov @ weights).item()
        std_per_asset = cov.diag().sqrt().numpy().tolist()
        return std, std_per_asset
        
    def evaluate(self,
                 test_data: pandas.DataFrame,
                 weights_dict: Dict[str, List[float] | List[str]],
                 cov_dict: Dict[str, List[float] | List[str]],
                 start_date: dt.datetime | pd.Timestamp,
                 test_start_date: dt.datetime | pd.Timestamp,
                 end_date: dt.datetime | pd.Timestamp
                 ) -> NumericResults:
                
        # Prepare the weights
        assets = list(weights_dict.keys())
        weights = list(weights_dict.values())
        weights = to_torch_tensor(weights)
        cov = cov_dict["cov"]
        cov = to_torch_tensor(cov)
        
        # Prepare the data
        test_data = test_data[assets]
        test_data = to_torch_tensor(test_data.to_numpy())
        
        # Get the strategy returns
        aggregate_return, aggregate_return_per_asset = (
            NumericEvaluator.__get_returns(test_data=test_data, weights=weights))
        std, std_per_asset = (
            NumericEvaluator.__get_standard_devs(weights=weights, cov=cov))
        
        # Create the result
        result = NumericResults(
            strategy_id=self.strategy_id,
            start_date=start_date,
            test_start_date=test_start_date,
            end_date=end_date,
            assets=assets,
            return_per_asset=aggregate_return_per_asset,
            return_=aggregate_return,
            std_per_asset=std_per_asset,
            std=std)
        
        return result
        
        
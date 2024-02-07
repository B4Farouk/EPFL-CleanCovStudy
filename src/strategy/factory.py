from strategy.base import Strategy
from strategy.strategies import MeanVarianceStrategy

from typing import Dict, Any

def get_strategy(name: str, cfg: Dict[str, Any]) -> Strategy:
    strategy_object = None
    
    match name:
        case "mean-variance":
            strategy_object = MeanVarianceStrategy(**cfg)
        case _:
            raise ValueError(f"Strategy {name} is invalid.")
        
    return strategy_object
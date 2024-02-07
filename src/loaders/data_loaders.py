import os
import pathlib
from typing import Dict, List, Tuple, Any, Optional, Iterable

import dask.dataframe
import dask.dataframe as dd
import pandas
import pandas as pd
import numpy as np

from local_dask import LocalDaskManager

class DataLoader:
    """Data Loader."""
    def __init__(self, config: Dict[str, Any], use_dask: bool =False):
        self.config = config
        self.loaded = False
        self.assets = None
        self.data   = None
        self.use_dask = use_dask

    @staticmethod
    def _load_assets(
        data_path: str | pathlib.Path,
        target_assets: Optional[str] =None) -> list[str]:
        
        found_assets = None
        with os.scandir(data_path) as scan_obj:
            found_assets = [entry.name for entry in scan_obj if entry.is_dir()]
            
        assets = None
        if target_assets is not None:
            if isinstance(target_assets, str):
                target_assets = target_assets.strip().split(",")
            elif isinstance(target_assets, Iterable[str]):
                target_assets = [asset.strip() for asset in target_assets]
            else:
                raise ValueError(f"Invalid type for target_assets: {type(target_assets)}")
            
            target_assets_set = set(target_assets)
            found_assets_set  = set(found_assets)
            assert target_assets_set.issubset(found_assets_set), (
                f"Target assets {target_assets_set} not a subset of the set of found assets {found_assets_set}"
            )
            assets = list(found_assets_set.intersection(target_assets_set))
        else:
            assets = found_assets
         
        return assets
    
    @staticmethod
    def _load(
        data_path: str | pathlib.Path,
        target_assets: Optional[str] =None,
        use_dask: bool =True) -> Tuple[List[str], Dict[str, dask.dataframe.DataFrame | pandas.DataFrame]]:
        if use_dask:
            LocalDaskManager.get_or_create_cluster()
            LocalDaskManager.get_or_create_client()
        
        # load the assets
        assets = DataLoader._load_assets(data_path=data_path, target_assets=target_assets)
        assert len(assets) > 0, "No assets found."
        assets_set = set(assets)
        
        # load the data
        data = {}
        with os.scandir(data_path) as outer:
            for outer_entry in outer:
                if not(outer_entry.is_dir()) or not(outer_entry.name in assets_set):
                    continue
                    
                with os.scandir(outer_entry) as inner:
                    for inner_entry in inner:
                        is_parquet_gzip = inner_entry.name.endswith(".parquet.gzip")
                        is_parquet      = inner_entry.name.endswith(".parquet")
                        
                        if not(is_parquet or is_parquet_gzip):
                            continue
                        
                        parquet_path = inner_entry.path
                        if use_dask:
                            data[outer_entry.name] = dd.read_parquet(parquet_path, engine="fastparquet")
                        else:
                            data[outer_entry.name] = pd.read_parquet(parquet_path, engine="fastparquet")
        
        # sanity checks
        assert len(data) == len(assets), (
            f"Number of assets {len(assets)} does not match number of loaded dataframes {len(data)}."
        )
        assert data.keys() == assets_set, (
            f"Loaded dataframes {data.keys()} do not match assets {assets_set}."
        )
                     
        return assets, data
        
    def _force_load(self):
        if self.loaded:
            assert self.assets is not None and self.data is not None
            return
        
        assets, data = DataLoader._load(
            data_path=self.config["data_path"],
            target_assets=self.config.get("assets", None),
            use_dask=self.use_dask)
        self.assets = assets
        self.data   = data
        self.loaded = True
         
    def load(self):
        self._force_load()
        return self.data
    
    def as_merged_dataframe(
        self,
        keep_cols: Optional[List[str] | str] =None,
        datetime_index: bool =True,
        return_asset_to_column_map: bool =False)\
        -> Tuple[pandas.DataFrame | dask.dataframe.DataFrame, Optional[Dict[str, int]]]:
        
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        elif not(isinstance(keep_cols, list)):
            raise ValueError(f"Invalid type for keep_cols: {type(keep_cols)}. Must be str or list.")
        
        if return_asset_to_column_map:
            assert keep_cols is not None and len(keep_cols) == 1
            
        self._force_load()
        
        result = None
        col_map = {} if return_asset_to_column_map else None
        for n, (asset, df) in enumerate(self.data.items()):
            if return_asset_to_column_map:
                col_map[asset] = n
            
            if keep_cols is not None:
                df = df[keep_cols]
            
            if n == 0:
                result = df.rename(columns={col: f"{col}_{asset}" for col in df.columns})
            else:
                df = df.rename(columns={col: f"{col}_{asset}" for col in df.columns})
                result = result.join(df, how="outer")
                                
        result = result.compute() if self.use_dask else result
        
        if datetime_index:
            result.index = pd.to_datetime(result.index)
        
        # Replace any occurence of infinity by zero
        result = result.replace([np.infty, -np.infty], 0)
        
        # Since we are using outer join, we will have NaNs
        # interpolate and fill forward and backward to fill the NaNs
        result = result.sort_index(ascending=True)
        result = result.interpolate(method="linear")
        result = result.ffill()
        
        return result, col_map
        
    def __str__(self) -> str:
        info = {
            "class": type(self).__name__,
            "state": "loaded" if self.loaded else "not loaded"
            }
        
        if self.loaded:
            info["assets"] = {
                "count": len(self.assets),
                "list": self.assets
            }
            
        return str(info)
        
    def __repr__(self) -> str:
        return str(self)
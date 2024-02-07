import os
import tarfile
import pathlib
import re

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import xlrd
import shutil


def process_raw_tar(
    data_path: str | pathlib.Path,
    save_path: Optional[str] =None,
    target_ext: str = [".csv", ".csv.gz"],
    ignore_file_pattern: Optional[str] =None,
    ignore_dir_pattern: Optional[str] =None,
    compression: str= "gzip",
    return_dataframes: bool =False
    ) -> List[pd.DataFrame]:
    
    
    def process_bbo_df(
        df: pd.DataFrame,
        sampling_freq: str="60T", # default is 60 minutes 
        ) -> pd.DataFrame:
        
        # convert the xltime to datetime object
        df["date"] = [xlrd.xldate_as_datetime(xltime, 0) for xltime in df["xltime"]]
        df.drop(columns=["xltime"], inplace=True)
        df.set_index("date", inplace=True)
        
        # select the columns we want
        df.rename(columns={"bid-price": "bidp", "ask-price": "askp"}, inplace=True)
        df = df[["bidp", "askp"]]
        
        # replace every occurence of "()" with NaN
        df = df.replace("()", pd.NA)
        
        # convert the columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # fill the missing values
        df = df.interpolate(method="linear")
        df = df.ffill()
        df = df.bfill()
        
        # resampling
        df = df.resample(sampling_freq).mean()
        
        # add a column for the medium price
        df["midp"] = df[["bidp", "askp"]].mean(axis=1)
        
        # add return column
        df["ret"] = (df["midp"] - df["midp"].shift(1)) / df["midp"].shift(1)
        df["ret"] = df["ret"].fillna(0)
        
        # add log return column
        df["log_ret"] = df["ret"].apply(lambda ret: np.log(1 + ret))
        
        # drop the bid and ask prices
        df.drop(columns=["bidp", "askp"], inplace=True)
        
        return df
        
    
    def crawl_and_process_tar(
        tar: tarfile.TarFile,
        target_ext: Tuple[str],
        save_path: Optional[str] =None,
        return_dataframes: bool =True,
        ignore_file_pattern: Optional[re.Pattern] =None,
        ignore_dir_pattern: Optional[re.Pattern] =None,
        compression: str= "gzip",
        recursive_depth: int =0) -> List[pd.DataFrame]:
        assert recursive_depth < 2, "recursive_depth must be less than 2, otherwise the function could missbehave..."
        
        if not(return_dataframes):
            assert save_path is not None, "save_path must be specified if return_dataframes is False."
        
        returned_dfs = [] if return_dataframes else None
        
        tar_infos = [
            member for member in tar.getmembers()
            if member.isfile() and member.name.endswith(target_ext)
            ]
            
        for tar_info in tar_infos:
            if tar_info.name.endswith(".tar"):
                # make a recursive call to process the inner tar file
                # unless it matches the ignore pattern
                
                if ignore_dir_pattern is not None:
                    matched = ignore_dir_pattern.match(tar_info.name)
                    if matched is not None:
                        continue
                
                concat_df = None
                with tar.extractfile(tar_info) as extracted_fobj:
                    with tarfile.open(fileobj=extracted_fobj, mode='r') as inner_tar:
                        clean_tar_name = (
                            tar_info.name
                            .replace(".tar", '')
                            .replace("./", '')
                            .replace(".", '-')
                        )
                        save_path_augmented = (
                            os.path.join(save_path, clean_tar_name)
                            if save_path is not None
                            else None
                            )
                        
                        dfs = crawl_and_process_tar(
                            tar=inner_tar,
                            target_ext=target_ext,
                            save_path=None, 
                            return_dataframes=True, # need to collect the dataframes from the inner tar
                            ignore_file_pattern=ignore_file_pattern,
                            ignore_dir_pattern=ignore_dir_pattern,
                            compression=compression,
                            recursive_depth=recursive_depth + 1)
                        
                        n_dfs = len(dfs)
                        if n_dfs > 1:
                            concat_df = pd.concat([df for df in dfs])
                        elif n_dfs == 1:
                            concat_df = dfs[0]
                        
                        if concat_df is not None:
                            concat_df.sort_index(inplace=True)
                        
                        # save to parquet
                        if save_path_augmented is not None and concat_df is not None:
                            os.makedirs(os.path.dirname(save_path_augmented), exist_ok=True)        
                            concat_df.to_parquet(
                                save_path_augmented+".parquet"+f".{compression}", engine="fastparquet", compression=compression)
                            print("saved:", save_path_augmented)
                
                if concat_df is not None and return_dataframes:
                    returned_dfs.append(concat_df)
                
            else:
                # process the file unless it matches the ignore pattern
                
                if ignore_file_pattern is not None:
                    matched = ignore_file_pattern.match(tar_info.name)
                    if matched is not None:
                        continue
                
                df = None
                with tar.extractfile(tar_info) as extracted_fobj:                  
                    try:
                        # read and process the file data  
                        df = pd.read_csv(
                            extracted_fobj, compression="gzip" if tar_info.name.endswith(".gz") else None,
                            encoding="utf-8", encoding_errors="strict")
                        df = process_bbo_df(df)    
                    except Exception:
                        err_file_path = os.path.join(save_path, tar_info.name)
                        print("Error in processing file: ", err_file_path)
                
                if df is not None:
                    returned_dfs.append(df)
        
        return returned_dfs
    
    
    data_path = pathlib.Path(data_path) if isinstance(data_path, str) else data_path
    if not os.path.exists(data_path):
        raise ValueError(f"Invalid path: {data_path}")
        
    # compile the regex patterns
    ignore_dir_pattern = (
        re.compile(ignore_dir_pattern, flags=re.IGNORECASE)
        if ignore_dir_pattern is not None
        else None
    )
    ignore_file_pattern = (
        re.compile(ignore_file_pattern, flags=re.IGNORECASE)
        if ignore_file_pattern is not None
        else None)
    
    # start processing the tar file
    clean_dataframes = []
    target_ext.append(".tar") # we want the processing to be hierarchical
    target_ext = tuple(target_ext)
    with tarfile.open(data_path, 'r') as tar:
        clean_dataframes = crawl_and_process_tar(
            tar=tar,
            target_ext=target_ext,
            save_path=save_path,
            return_dataframes=return_dataframes,
            ignore_file_pattern=ignore_file_pattern,
            ignore_dir_pattern=ignore_dir_pattern,
            compression=compression)
    
    return clean_dataframes


def unify_parquet_names(
    data_path: str | pathlib.Path,
    unique_name: str,
    save_path: Optional[str | pathlib.Path] =None):
    
    with os.scandir(data_path) as outer:
        for outer_entry in outer:
            if not(outer_entry.is_dir()):
                continue
            
            with os.scandir(outer_entry.path) as inner:
                for inner_entry in inner:                
                    is_parquet_gzip = (
                        inner_entry.name.endswith(".parquet.gzip")
                        or inner_entry.name.endswith(".parquet.gz"))
                    is_parquet = inner_entry.name.endswith(".parquet")
                    
                    if not(is_parquet or is_parquet_gzip):
                        continue
                    
                    new_filename = (
                        f"{unique_name}.parquet.gzip"
                        if is_parquet_gzip
                        else f"{unique_name}.parquet"
                    )
                    
                    if save_path is not None:
                        old_filepath = inner_entry.path
                        
                        new_dirpath = os.path.join(save_path, outer_entry.name.split('-')[0])
                        os.makedirs(new_dirpath, exist_ok=True)
                        new_filepath = os.path.join(new_dirpath, new_filename)
                        
                        shutil.copyfile(old_filepath, new_filepath)
                    
                    else:
                        old_filepath = os.path.join(outer_entry.path, inner_entry.name)
                        
                        new_dirpath = outer_entry.path.split('-')[0]
                        os.makedirs(new_dirpath, exist_ok=True)
                        new_filepath = os.path.join(new_dirpath, new_filename)
                        
                        os.rename(old_filepath, new_filepath)
                        

                    
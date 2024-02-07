import pathlib
import sys
src_path = pathlib.Path("./src").absolute()
if src_path not in sys.path:
    sys.path.append(str(src_path))

import argparse

import time

from preprocessing import *

##########################
# Tests
##########################

def test_process_raw_tar():
    print("> test_process_raw_tar...")
    
    test_args = {
        "data_path" : "./data/raw/sp100_2004_2008.tar",
        "save_path": "./debug-results/data/processed/",
        "target_ext" : [".csv", ".csv.gz"],
        "ignore_file_pattern": r"(.*(trade).*)|(.*(2004).*)",
        "ignore_dir_pattern": r"(.*(trade).*)|(.*(2004).*)",
        "compression": "gzip",
        "return_dataframes": False
    }
    exception = None
    
    start_time = None
    end_time = None
    try:
        start_time = time.time()
        process_raw_tar(**test_args)
        end_time = time.time()
    except Exception as e:
        passed_or_failed = "FAILED"
        exception = e
            
    if exception is None:
        passed_or_failed = "PASSED"
        
    if start_time is not None and end_time is not None:
        print(f"time: {end_time - start_time} seconds")
    
    print(f"< test_process_raw_tar: {passed_or_failed}")
    
    if exception is not None:
        raise exception
    

def test_unify_parquet_names():
    print("> test_process_raw_tar...")
    
    test_args = {
        "data_path" : "./data/processed/v00/",
        "unique_name": "data_v01",
        "save_path": "./debug-results/data/processed/v01/"
    }
    exception = None
    
    start_time = None
    end_time = None
    try:
        start_time = time.time()
        unify_parquet_names(**test_args)
        end_time = time.time()
    except Exception as e:
        passed_or_failed = "FAILED"
        exception = e
            
    if exception is None:
        passed_or_failed = "PASSED"
        
    if start_time is not None and end_time is not None:
        print(f"time: {end_time - start_time} seconds")
    
    print(f"< test_process_raw_tar: {passed_or_failed}")
    
    if exception is not None:
        raise exception


__ALL_TESTS = [
    test_process_raw_tar,
    test_unify_parquet_names
]

##########################
# Test runner
##########################

def test(**kwargs):
    
    print("Running tests...")
    
    if kwargs.get("all", False):
        for test_fn in __ALL_TESTS:
            test_fn()
    else:
        if kwargs.get("process_raw_tar_no_save", False):
            test_process_raw_tar()
        if kwargs.get("unify_parquet_names", False):
            test_unify_parquet_names()
    
    print("All tests passed!")

##########################
# Main
##########################

if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser(description="Testing script.")
    
        parser.add_argument("--process_raw_tar", action='store_true', default=False, help="Test the process_raw_tar function.")
        parser.add_argument("--unify_parquet_names", action='store_true', default=False, help="Test the unify_parquet_names function.")
        parser.add_argument("--all", action="store_true", default=False, help="Run all tests.")
    
        return parser.parse_args()

    args = get_args()
    kwargs = vars(args)
    test(**kwargs)
    
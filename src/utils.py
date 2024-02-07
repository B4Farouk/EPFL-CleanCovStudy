import logging
import pathlib
import sys
import os
from typing import Optional, List

from io import TextIOWrapper

import numpy
import torch

##################################
# Functions
##################################

def to_torch_tensor(
    array: torch.Tensor | numpy.ndarray | List[float],
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    requires_grad: bool =False):
    
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, list):
        tensor = torch.tensor(array)
    else:
        tensor = array
    
    tensor = tensor.to(dtype)
    tensor = tensor.requires_grad_(requires_grad)
    
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        tensor = tensor.to(device)
    elif device is None and torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda"))
    # else normally it must already be on cpu
     
    return tensor

def get_or_create_logger(
    name: Optional[str] =None,
    path_or_io_wrapper: Optional[str | pathlib.Path | TextIOWrapper] =None,
    level: Optional[str] =None) -> logging.Logger:
    
    is_none = path_or_io_wrapper is None
    is_path = isinstance(path_or_io_wrapper, (str, pathlib.Path))
    is_io_wrapper = isinstance(path_or_io_wrapper, TextIOWrapper)
    assert is_none or is_path or is_io_wrapper,\
        f"path_or_io_wrapper must be either a path (str or pathlib.Path) or a TextIOWrapper, got {type(path_or_io_wrapper)} instead."

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    
    # If the logger already has a handler that writes to the same file, do nothing
    path = path_or_io_wrapper if is_path else None
    io_wrapper = path_or_io_wrapper if is_io_wrapper else None
    if not(is_none) and any(
        [(isinstance(handler, logging.FileHandler) and handler.baseFilename == (path if is_path else io_wrapper.name))
         for handler in logger.handlers]):
        return logger   

    # otherwise add a new handler that writes to the file specified by path
    # create stream handler
    stream_handler = None
    if is_path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        io_wrapper = open(path, mode='w')
        stream_handler = logging.StreamHandler(io_wrapper)
    elif is_io_wrapper:
        stream_handler = logging.StreamHandler(io_wrapper)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
    
    # create formatter and add it to the handler
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(formatter)
    
    # add the handler to the logger
    logger.addHandler(stream_handler)    
    return logger

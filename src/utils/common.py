import os
import random
import numpy as np
import torch
import logging
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Safe loading of YAML configuration files."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Sets up a logger that writes to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler (optional)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def seed_everything(seed: int = 1234):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CuDNN (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Global seed set to: {seed}")

def get_device(device_config: str) -> torch.device:
    """Returns the torch device."""
    if device_config == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_config == "mps" and torch.backends.mps.is_available():
         # Support for Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")
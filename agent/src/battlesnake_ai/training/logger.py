import logging
import os
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    SummaryWriter = None  # type: ignore[misc, assignment]
    HAS_TENSORBOARD = False

def setup_logger(log_dir: str = "logs", log_name: str = "training") -> logging.Logger:
    """
    Setup standard Python logging to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

def get_tensorboard_writer(log_dir: str = "logs/tensorboard", run_name: str = None) -> 'SummaryWriter':
    """
    Get a TensorBoard SummaryWriter instance for tracking metrics.
    """
    if not HAS_TENSORBOARD:
        print("Warning: TensorBoard not installed. Cannot use SummaryWriter.")
        return None
        
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        
    full_path = os.path.join(log_dir, run_name)
    os.makedirs(full_path, exist_ok=True)
    return SummaryWriter(log_dir=full_path)

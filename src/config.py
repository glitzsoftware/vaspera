from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Model directory
MODEL_DIR = ROOT_DIR / "models"

# Default parameters
RANDOM_SEED = 42

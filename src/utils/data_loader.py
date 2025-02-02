import pandas as pd
from pathlib import Path
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw data from the raw data directory."""
    file_path = RAW_DATA_DIR / filename
    return pd.read_csv(file_path)

def save_processed_data(data: pd.DataFrame, filename: str) -> None:
    """Save processed data to the processed data directory."""
    file_path = PROCESSED_DATA_DIR / filename
    data.to_csv(file_path, index=False)

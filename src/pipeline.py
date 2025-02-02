from utils.data_loader import load_raw_data, save_processed_data
from utils.model_utils import split_data, evaluate_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(input_file: str, output_file: str):
    """Run the complete data processing and modeling pipeline."""
    try:
        # Load data
        logger.info("Loading raw data...")
        data = load_raw_data(input_file)
        
        # Add your processing steps here
        
        # Save processed data
        logger.info("Saving processed data...")
        save_processed_data(data, output_file)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

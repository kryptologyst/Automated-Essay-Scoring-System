import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config(BaseModel):
    """Configuration settings for the essay scoring system."""
    
    # Model settings
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    # Database settings
    database_url: str = "sqlite:///essay_scoring.db"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # File paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    # Scoring settings
    min_score: float = 0.0
    max_score: float = 10.0
    score_precision: int = 2
    
    class Config:
        env_file = ".env"

# Create configuration instance
config = Config()

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    config.logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'essay_scoring.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Create necessary directories
for directory in [config.data_dir, config.models_dir, config.logs_dir]:
    directory.mkdir(exist_ok=True)

logger.info("Configuration loaded successfully")

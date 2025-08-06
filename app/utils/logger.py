import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def get_logger(name: str):
    """
    Returns a configured logger instance.
    """
    return logging.getLogger(name)
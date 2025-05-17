import logging
from pathlib import Path

# Determine the project root directory, assuming this file is in game_of_thrones_sim/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Go up three levels: utils -> game_of_thrones_sim -> project root

LOG_FILE_NAME = "got_sim_gui_v3.log"
LOG_FILE_PATH = PROJECT_ROOT / LOG_FILE_NAME # Place log file in the project root

def setup_logging():
    """Configures the global logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(threadName)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=LOG_FILE_PATH,
        filemode='w'
    )
    # Consistent logger name
    log = logging.getLogger("got-sim-v3")

    # Suppress noisy logs from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("google.ai").setLevel(logging.INFO) # Allow some info from google-ai
    logging.getLogger("google.api_core").setLevel(logging.INFO)

    # Add a console handler to also print INFO and above to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s') # Simpler format for console
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler) # Add to root logger

    log.info(f"Logging initialized. Log file: {LOG_FILE_PATH}")
    return log

# Initialize and export the logger instance
log = setup_logging()

if __name__ == '__main__':
    # Example usage if you run this file directly
    log.info("Logger test: Info message.")
    log.warning("Logger test: Warning message.")
    log.error("Logger test: Error message.")
    log.debug("Logger test: Debug message (won't show with INFO level).")

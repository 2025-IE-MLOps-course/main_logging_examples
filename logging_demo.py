# %% Step 1: Logging vs Print

import os
import yaml
import logging

print("this is a refular print statement")

logging.basicConfig(level=logging.INFO)
logging.info("this is an info message")
logging.warning("this is a warning message")


# %% Step 2: Logging Levels

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # remove all handlers

logging.basicConfig(level=logging.CRITICAL)
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")


# %% Step 3: Logging to a File

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # remove all handlers

logging.basicConfig(
    filename="logs/simple.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("This log will appear in the logs/simple.log file.")

# %% Step 4: Using a YAML Config File for Logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # remove all handlers


# Load config from YAML file
with open("logging_config.yaml", "r") as f:
    log_cfg = yaml.safe_load(f)

# Ensure logs/ folder exists
log_dir = os.path.dirname(log_cfg["log_file"])
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Set up logging as per config
logging.basicConfig(
    filename=log_cfg["log_file"],
    level=getattr(logging, log_cfg["level"]),
    format=log_cfg["format"],
    datefmt=log_cfg["datefmt"],
)
logging.info("Logging configured from YAML file!")
logging.warning("This is a warning using YAML-configured logging.")

# %% Step 5: Best Practice—Reusable Logging Setup Function

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # remove all handlers


def setup_logging(logging_config: dict):
    """
    Set up logging based on a configuration dictionary.
    Reads options like log file, level, and formatting from the config.
    """
    # Extract log file path from config (default if missing)
    log_file = logging_config.get("log_file", "logs/default.log")

    # Get directory part of the log file path
    log_dir = os.path.dirname(log_file)

    # Create the log directory if it does not exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure the root logger: file, level, format, date format
    logging.basicConfig(
        filename=log_file,  # Log file path
        # Log level as string (e.g., "INFO" → logging.INFO)
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get(
            "format", "%(asctime)s - %(levelname)s - %(message)s"),  # Message format
        datefmt=logging_config.get(
            "datefmt", "%Y-%m-%d %H:%M:%S")   # Date/time format
    )

    # Add an additional handler for logging to the console (not just to file)
    console = logging.StreamHandler()  # Console handler
    # Set console handler log level (same as file, from config)
    console.setLevel(getattr(logging, logging_config.get("level", "INFO")))
    # Create a formatter for console messages (same format as file)
    formatter = logging.Formatter(
        logging_config.get(
            "format", "%(asctime)s - %(levelname)s - %(message)s"),
        datefmt=logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    )
    # Attach the formatter to the console handler
    console.setFormatter(formatter)
    # Add the console handler to the root logger so logs go to both file and console
    logging.getLogger().addHandler(console)


# Load logging configuration from a YAML file
with open("logging_config.yaml", "r") as f:
    log_cfg = yaml.safe_load(f)   # Read YAML config as a dict

# Call our reusable setup function with the loaded config
setup_logging(log_cfg)

# Log messages of various levels (these go to both console and file)
logging.info("This log uses our reusable setup function from YAML config!")
logging.warning("This warning also uses the config-driven logger.")

# %%

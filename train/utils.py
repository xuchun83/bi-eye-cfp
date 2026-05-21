import json
import logging
import os


def save_summary_to_json(summary, output_file):
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)


def setup_logger(log_dir, log_file="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

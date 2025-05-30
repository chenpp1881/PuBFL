import torch
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from transformers import AutoTokenizer, T5EncoderModel

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def setup_logger(current_time):
    logger = logging.getLogger("MyApp")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file_name = f"logs/test_{current_time}.log"

    file_handler = RotatingFileHandler(
        log_file_name, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


CONFIG = {
    "private_dataset_name_list": ["IO", "timestamp", "reentrancy"],
    "private_data_len_list": [4268, 1522, 2764],
    "public_dataset_length": 10000,
    "private_net_name_list": ["graphcodebert", "codet5", "unixcoder"],
    "num_classes": 2,
    "test_size": 0.2,
    "random_seed": 42,
    "batch_size": 64,
    "pretrain_learning_rate": 1e-6,
    "public_learning_rates": [1e-6, 1e-6, 1e-6],
    "private_learning_rates": [1e-6, 1e-6, 1e-6],
    "pretrain_epochs": 30,
    "public_epochs": 1,
    "private_epochs": 1,
    "communication_epochs": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "identifiers": {
        "graphcodebert": "microsoft/graphcodebert-base",
        "codet5": "Salesforce/codet5-base",
        "unixcoder": "microsoft/unixcoder-base"
    },
    "tokenizers": {
        "graphcodebert": AutoTokenizer.from_pretrained("microsoft/graphcodebert-base"),
        "codet5": AutoTokenizer.from_pretrained("Salesforce/codet5-base"),
        "unixcoder": AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    },
    "max_seq_length": 512,
    "preprocess": True,
    "logger": setup_logger(current_time),
}

logger = CONFIG['logger']
tokenizers = CONFIG['tokenizers']

logger.info("--- Start Configuration ---")
for key, value in CONFIG.items():
    if key == 'tokenizers' or key == 'logger': continue
    logger.info(f"{key}: {repr(value)}")
logger.info("--- End Configuration ---")

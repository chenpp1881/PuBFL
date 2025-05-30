import os
import re
import math
import json
import torch
import random
from config import *
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PublicDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        code = self.data[idx]

        encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0)
        }


class PrivateDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        code = item['code']
        label = item['label']

        encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def clean_solidity_code(source_code: str) -> str:
    source_code = re.sub(r'/\*[\s\S]*?\*/', '', source_code)
    source_code = re.sub(r'//.*', '', source_code)
    lines = source_code.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line == ';':
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def get_IO(path='Datasets/IO/dataset.json', preprocess=CONFIG['preprocess']):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if preprocess:
        for i in data:
            i['code'] = clean_solidity_code(i['code'])

    return data


def get_reentrancy(path='Datasets/reentrancy/data.json', preprocess=CONFIG['preprocess']):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    formed_data = []
    for key, val in data.items():
        for k, v in val.items():
            formed_data.append({'code': v['code'], 'label': v['lable']})

    if preprocess:
        for i in formed_data:
            i['code'] = clean_solidity_code(i['code'])

    return formed_data


def get_timestamp(path='Datasets/timestamp/data.json', preprocess=CONFIG['preprocess']):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    formed_data = []
    for key, val in data.items():
        for k, v in val.items():
            formed_data.append({'code': v['code'], 'label': v['lable']})

    if preprocess:
        for i in formed_data:
            i['code'] = clean_solidity_code(i['code'])

    return formed_data


def get_smartbugs(path='Datasets/SmartBugs/data.jsonl', preprocess=CONFIG['preprocess']):
    formed_data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            formed_data.append(data['contract'])

    if preprocess:
        for idx in range(len(formed_data)):
            formed_data[idx] = clean_solidity_code(formed_data[idx])

    return formed_data


def check(dataset):
    zero_count = 0
    one_count = 0
    for i in dataset:
        if i['label'] == 0:
            zero_count += 1
        if i['label'] == 1:
            one_count += 1
    return zero_count, one_count


def load_datasets_for_models(max_seq_length=512):
    logger.info("Loading original datasets...")
    IO_raw = get_IO()
    timestamp_raw = get_timestamp()
    reentrancy_raw = get_reentrancy()
    smartbugs_raw = get_smartbugs()
    logger.info(f"Original IO dataset size: {len(IO_raw)}")
    logger.info(f"Original timestamp dataset size: {len(timestamp_raw)}")
    logger.info(f"Original reentrancy dataset size: {len(reentrancy_raw)}")
    logger.info(f"smartbugs dataset size: {len(smartbugs_raw)}")

    logger.info("Splitting datasets...")
    IO_train_raw, IO_test_raw = train_test_split(
        IO_raw,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
        stratify=[item['label'] for item in IO_raw]
    )

    timestamp_train_raw, timestamp_test_raw = train_test_split(
        timestamp_raw,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
        stratify=[item['label'] for item in timestamp_raw]
    )

    reentrancy_train_raw, reentrancy_test_raw = train_test_split(
        reentrancy_raw,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
        stratify=[item['label'] for item in reentrancy_raw]
    )
    logger.info("Datasets split.")

    all_datasets = {}

    for name, tokenizer_instance in tokenizers.items():
        logger.info(f"Creating datasets for tokenizer: {name}")
        datasets_for_tokenizer = {}

        datasets_for_tokenizer['smartbugs'] = PublicDataset(
            smartbugs_raw,
            tokenizer_instance,
            max_seq_length
        )

        datasets_for_tokenizer['IO_train'] = PrivateDataset(
            IO_train_raw,
            tokenizer_instance,
            max_seq_length
        )
        datasets_for_tokenizer['IO_test'] = PrivateDataset(
            IO_test_raw,
            tokenizer_instance,
            max_seq_length
        )

        datasets_for_tokenizer['timestamp_train'] = PrivateDataset(
            timestamp_train_raw,
            tokenizer_instance,
            max_seq_length
        )
        datasets_for_tokenizer['timestamp_test'] = PrivateDataset(
            timestamp_test_raw,
            tokenizer_instance,
            max_seq_length
        )

        datasets_for_tokenizer['reentrancy_train'] = PrivateDataset(
            reentrancy_train_raw,
            tokenizer_instance,
            max_seq_length
        )
        datasets_for_tokenizer['reentrancy_test'] = PrivateDataset(
            reentrancy_test_raw,
            tokenizer_instance,
            max_seq_length
        )

        all_datasets[name] = datasets_for_tokenizer

        for ds_name, dataset_obj in datasets_for_tokenizer.items():
            logger.info(f"  Tokenizer: {name}, Dataset: {ds_name}, Samples: {len(dataset_obj)}")
            sample = dataset_obj[0]
            log_msg = f"    Sample shapes: input_ids={sample['input_ids'].shape}, attention_mask={sample['attention_mask'].shape}"
            if 'label' in sample:
                log_msg += f", label={sample['label'].shape}"
            logger.info(log_msg)
    return all_datasets


def get_random_subset(dataset, size):
    total_samples = len(dataset)
    shuffled_indices = torch.randperm(total_samples)
    selected_indices = shuffled_indices[:size]
    random_subset = Subset(dataset, selected_indices)

    return random_subset


def get_dataloader(name, size, train_dataset, test_dataset, train_batch_size, test_batch_size, logger):
    if size != 0:
        train_dataset = get_random_subset(train_dataset, size)

    train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=64, pin_memory=True)
    for batch_idx, batch in enumerate(train_dl):
        input_data = batch['input_ids']
        target_data = batch.get('label')

        log_message = f"{name} 训练集大小：{len(train_dataset)}，{name} 训练集批次 {batch_idx}: 数据形状 {input_data.shape}"
        if target_data is not None:
            log_message += f"，标签形状 {target_data.shape}"
        else:
            log_message += "，无标签\n"
        logger.info(log_message)
        break

    test_dl = None
    if test_dataset:
        test_dl = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=64, pin_memory=True)
        for batch_idx, batch in enumerate(test_dl):
            input_data = batch['input_ids']
            target_data = batch.get('label')

            log_message = f"测试集大小：{len(test_dataset)}, {name} 测试集批次 {batch_idx}: 数据形状 {input_data.shape}"
            if target_data is not None:
                log_message += f"，标签形状 {target_data.shape}\n"
            else:
                log_message += "，无标签\n"
            logger.info(log_message)
            break
    return train_dl, test_dl


results = load_datasets_for_models()
logger.info(f"Preprocess: {CONFIG['preprocess']}")

all_dataloaders = {}
for tokenizer_name, datasets_dict in results.items():
    logger.info(f"Creating DataLoaders for tokenizer: {tokenizer_name}")
    dataloaders_for_tokenizer = {}
    dataloaders_for_tokenizer['IO_train_dl'] = DataLoader(
        datasets_dict['IO_train'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['IO_test_dl'] = DataLoader(
        datasets_dict['IO_test'],
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['timestamp_train_dl'] = DataLoader(
        datasets_dict['timestamp_train'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['timestamp_test_dl'] = DataLoader(
        datasets_dict['timestamp_test'],
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['reentrancy_train_dl'] = DataLoader(
        datasets_dict['reentrancy_train'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['reentrancy_test_dl'] = DataLoader(
        datasets_dict['reentrancy_test'],
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=64,
        pin_memory=True
    )
    dataloaders_for_tokenizer['smartbugs_dl'] = DataLoader(
        get_random_subset(datasets_dict['smartbugs'], size=CONFIG['public_dataset_length']),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )
    all_dataloaders[tokenizer_name] = dataloaders_for_tokenizer

logger.info("\n--- Checking DataLoader Batch Shapes ---")
for tokenizer_name, dataloaders_dict in all_dataloaders.items():
    logger.info(f"--- Tokenizer: {tokenizer_name} ---")
    for dl_name, dataloader in dataloaders_dict.items():
        try:
            first_batch = next(iter(dataloader))

            input_ids = first_batch.get('input_ids')
            attention_mask = first_batch.get('attention_mask')
            labels = first_batch.get('label')

            log_message = f"  DataLoader: {dl_name}"
            if input_ids is not None:
                log_message += f", input_ids shape: {input_ids.shape}"
            if attention_mask is not None:
                log_message += f", attention_mask shape: {attention_mask.shape}"
            if labels is not None:
                log_message += f", label shape: {labels.shape}"
            else:
                if 'label' not in first_batch:
                    log_message += ", no label (as expected for PublicDataset)"
                else:
                    log_message += ", label key exists but is None?"

            logger.info(log_message)

        except StopIteration:
            logger.warning(f"  DataLoader: {dl_name} is empty.")
        except Exception as e:
            logger.error(f"  Error checking DataLoader {dl_name} for tokenizer {tokenizer_name}: {e}")

graphcodebert_smartbugs_dl = all_dataloaders['graphcodebert']['smartbugs_dl']
codet5_smartbugs_dl = all_dataloaders['codet5']['smartbugs_dl']
unixcoder_smartbugs_dl = all_dataloaders['unixcoder']['smartbugs_dl']

graphcodebert_IO_train_dl = all_dataloaders['graphcodebert']['IO_train_dl']
codet5_IO_train_dl = all_dataloaders['codet5']['IO_train_dl']
unixcoder_IO_train_dl = all_dataloaders['unixcoder']['IO_train_dl']

graphcodebert_IO_test_dl = all_dataloaders['graphcodebert']['IO_test_dl']
codet5_IO_test_dl = all_dataloaders['codet5']['IO_test_dl']
unixcoder_IO_test_dl = all_dataloaders['unixcoder']['IO_test_dl']

graphcodebert_timestamp_train_dl = all_dataloaders['graphcodebert']['timestamp_train_dl']
codet5_timestamp_train_dl = all_dataloaders['codet5']['timestamp_train_dl']
unixcoder_timestamp_train_dl = all_dataloaders['unixcoder']['timestamp_train_dl']

graphcodebert_timestamp_test_dl = all_dataloaders['graphcodebert']['timestamp_test_dl']
codet5_timestamp_test_dl = all_dataloaders['codet5']['timestamp_test_dl']
unixcoder_timestamp_test_dl = all_dataloaders['unixcoder']['timestamp_test_dl']

graphcodebert_reentrancy_train_dl = all_dataloaders['graphcodebert']['reentrancy_train_dl']
codet5_reentrancy_train_dl = all_dataloaders['codet5']['reentrancy_train_dl']
unixcoder_reentrancy_train_dl = all_dataloaders['unixcoder']['reentrancy_train_dl']

graphcodebert_reentrancy_test_dl = all_dataloaders['graphcodebert']['reentrancy_test_dl']
codet5_reentrancy_test_dl = all_dataloaders['codet5']['reentrancy_test_dl']
unixcoder_reentrancy_test_dl = all_dataloaders['unixcoder']['reentrancy_test_dl']

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from load_dataset import *
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()


def train(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer, device: str, epoch_num: int,
          scheduler=None) -> float:
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"第 {epoch_num + 1} 轮训练", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('label')

        if input_ids is None or attention_mask is None or labels is None:
            logger.warning(f"Skipping batch {batch_idx} due to missing keys.")
            continue

        optimizer.zero_grad()

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        except AttributeError:
            logger.warning("Model output does not have .logits attribute, assuming direct logit output.")
            logits = outputs
        except Exception as e:
            logger.error(f"Error during model forward pass in train: {e}")
            continue

        try:
            if labels.dtype != torch.long: labels = labels.long()
            if labels.dim() > 1 and labels.shape[1] == 1: labels = labels.squeeze(1)
            if logits.shape[0] != labels.shape[0]:
                logger.warning(
                    f"Batch {batch_idx}: Logits batch size ({logits.shape[0]}) != Labels batch size ({labels.shape[0]}). Skipping.")
                continue

            loss = loss_fn(logits, labels)
        except Exception as e:
            logger.error(f"Error during loss calculation in train: {e}")
            continue

        if isinstance(loss, torch.Tensor) and loss.dim() > 0 and torch.cuda.device_count() > 1 and isinstance(model,
                                                                                                              nn.DataParallel):
            loss = loss.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss value (NaN or Inf) detected at batch {batch_idx}. Skipping update.")
            continue
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return avg_loss


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: str) -> Tuple[
    float, float, float, float, float]:
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="评估中", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            labels = batch.get('label')

            if input_ids is None or attention_mask is None or labels is None:
                logger.warning(f"Skipping batch {batch_idx} in eval due to missing keys.")
                continue

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            except AttributeError:
                logger.warning("Model output does not have .logits attribute, assuming direct logit output.")
                logits = outputs
            except Exception as e:
                logger.error(f"Error during model forward pass in evaluate: {e}")
                continue

            try:
                if labels.dtype != torch.long: labels = labels.long()
                if labels.dim() > 1 and labels.shape[1] == 1: labels = labels.squeeze(1)
                if logits.shape[0] != labels.shape[0]:
                    logger.warning(
                        f"Eval Batch {batch_idx}: Logits batch size ({logits.shape[0]}) != Labels batch size ({labels.shape[0]}). Skipping loss calc.")
                else:
                    loss = loss_fn(logits, labels)
                    if isinstance(loss,
                                  torch.Tensor) and loss.dim() > 0 and torch.cuda.device_count() > 1 and isinstance(
                        model, nn.DataParallel):
                        loss = loss.mean()
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                    else:
                        logger.warning(f"Invalid loss value (NaN or Inf) in eval batch {batch_idx}.")
            except Exception as e:
                logger.error(f"Error during loss calculation in evaluate: {e}")

            try:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logger.error(f"Error during prediction/label collection: {e}")

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    if not all_labels or not all_preds:
        logger.warning("Evaluation resulted in empty predictions or labels. Returning zero metrics.")
        return avg_loss, 0.0, 0.0, 0.0, 0.0
    try:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary',
                                                                   zero_division=0, labels=[0, 1])
    except Exception as e:
        logger.error(f"Error during metric calculation: {e}")
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return avg_loss, accuracy, precision, recall, f1


def pretrain(model, tokenizer, dataset_name, model_name, train_loader, test_loaders, epochs):
    logger.info(f"Pretain {model_name} on {dataset_name}...")

    model = nn.DataParallel(model).to(CONFIG['device'])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["pretrain_learning_rate"])

    test_f1s = []
    for test_loader in test_loaders:
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, CONFIG["device"])
        test_f1s.append(test_f1)
    logger.info(f"epoch 0/{CONFIG['pretrain_epochs']}, Test F1: {test_f1s}")

    all_train_loss, all_test_loss, all_test_acc, all_test_prec, all_test_rec, all_test_f1 = [], [], [], [], [], []
    for epoch in range(epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, CONFIG["device"], epoch)
        test_losses, test_accs, test_precs, test_recs, test_f1s = [], [], [], [], []
        for test_loader in test_loaders:
            test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, CONFIG["device"])
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_precs.append(test_prec)
            test_recs.append(test_rec)
            test_f1s.append(test_f1)

        all_train_loss.append(train_loss)
        all_test_loss.append(test_losses)
        all_test_acc.append(test_accs)
        all_test_prec.append(test_precs)
        all_test_rec.append(test_recs)
        all_test_f1.append(test_f1s)
        logger.info(f"epoch {epoch + 1}/{CONFIG['pretrain_epochs']}: \n"
                    f"Train Loss: {all_train_loss}\nTest Loss: {all_test_loss}\nTest Acc: {all_test_acc}\n"
                    f"Test Precision: {all_test_prec}\nTest Recall: {all_test_rec}\nTest F1: {all_test_f1}")

        save_directory = f'Pretrained/{current_time}/{dataset_name}_{model_name}'
        os.makedirs(save_directory, exist_ok=True)

        logger.info(f"Saving model and metrics to {save_directory}...")
        np.save(os.path.join(save_directory, "train_loss.npy"), np.array(all_train_loss))
        np.save(os.path.join(save_directory, "test_loss.npy"), np.array(all_test_loss))
        np.save(os.path.join(save_directory, "test_acc.npy"), np.array(all_test_acc))
        np.save(os.path.join(save_directory, "test_pre.npy"), np.array(all_test_prec))
        np.save(os.path.join(save_directory, "test_rec.npy"), np.array(all_test_rec))
        np.save(os.path.join(save_directory, "test_f1.npy"), np.array(all_test_f1))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        logger.info("Model, tokenizer and metrics saved.")


def overall_evaluate(network_names, dataset_names, networks, test_dls):
    acc_epoch_list, precision_epoch_list, recall_epoch_list, f1_epoch_list = [], [], [], []
    for network, network_name, three_test_dl in zip(networks, network_names, test_dls):
        acc_each_list, precision_each_list, recall_each_list, f1_each_list = [], [], [], []
        for test_dl, dataset_name in zip(three_test_dl, dataset_names):
            logger.info(f"Evaluate {network_name} on {dataset_name}...")
            _, acc, precision, recall, f1 = evaluate(network, test_dl, nn.CrossEntropyLoss(), CONFIG['device'])
            acc_each_list.append(acc)
            precision_each_list.append(precision)
            recall_each_list.append(recall)
            f1_each_list.append(f1)
        acc_epoch_list.append(acc_each_list)
        precision_epoch_list.append(precision_each_list)
        recall_epoch_list.append(recall_each_list)
        f1_epoch_list.append(f1_each_list)
    return acc_epoch_list, precision_epoch_list, recall_epoch_list, f1_epoch_list


if __name__ == '__main__':
    net1 = AutoModelForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)
    pretrain(net1, CONFIG['tokenizers']['codet5'], 'IO', 'graphcodebert', graphcodebert_IO_train_dl,
             [graphcodebert_IO_test_dl, graphcodebert_timestamp_test_dl, graphcodebert_reentrancy_test_dl], 30)

    net2 = AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5-base", num_labels=2)
    pretrain(net2, CONFIG['tokenizers']['codet5'], 'timestamp', 'codet5', codet5_timestamp_train_dl,
             [codet5_IO_test_dl, codet5_timestamp_test_dl, codet5_reentrancy_test_dl], 30)

    net3 = AutoModelForSequenceClassification.from_pretrained("microsoft/unixcoder-base", num_labels=2)
    pretrain(net3, CONFIG['tokenizers']['codet5'], 'reentrancy', 'unixcoder', unixcoder_reentrancy_train_dl,
             [unixcoder_IO_test_dl, unixcoder_timestamp_test_dl, unixcoder_reentrancy_test_dl], 30)

import copy
import torch
import random
import numpy as np
from load_network import *
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import *
import math

base_dir = "Data"
target_dir = os.path.join(base_dir, current_time)
os.makedirs(target_dir, exist_ok=True)
device = CONFIG["device"]

random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if device == "cuda":
    torch.cuda.manual_seed_all(CONFIG["random_seed"])

private_dataset_name_list = CONFIG["private_dataset_name_list"]
private_data_len_list = CONFIG["private_data_len_list"]
public_dataset_length = CONFIG["public_dataset_length"]
private_net_name_list = CONFIG["private_net_name_list"]

train_dls = [graphcodebert_IO_train_dl, codet5_timestamp_train_dl, unixcoder_reentrancy_train_dl]
test_dls = [
    [graphcodebert_IO_test_dl, graphcodebert_timestamp_test_dl, graphcodebert_reentrancy_test_dl],
    [codet5_IO_test_dl, codet5_timestamp_test_dl, codet5_reentrancy_test_dl],
    [unixcoder_IO_test_dl, unixcoder_timestamp_test_dl, unixcoder_reentrancy_test_dl]
]

public_learning_rates = CONFIG['public_learning_rates']
private_learning_rates = CONFIG['private_learning_rates']
public_epochs = CONFIG['public_epochs']
private_epochs = CONFIG['private_epochs']
communication_epochs = CONFIG['communication_epochs']
batch_size = CONFIG['batch_size']
num_classes = CONFIG['num_classes']

para_lambda = 0.0051
para_mu = 0.02
initial_para_tau = 1.2
final_para_tau = 0.8
ema_decay = 0.999
grad_scale_alpha = 0.4
logger.info(f"public_learning_rates: {public_learning_rates}, private_learning_rates: {private_learning_rates}, "
            f"para_lambda: {para_lambda}, para_mu: {para_mu}, para_omega: {para_omega}, "
            f"initial_para_tau: {initial_para_tau}, final_para_tau: {final_para_tau}, "
            f"ema_decay: {ema_decay}, grad_scale_alpha: {grad_scale_alpha}")

logger.info("Prepare Models...")
networks = [
    AutoModelForSequenceClassification.from_pretrained("Pretrained/models/IO_graphcodebert"),
    AutoModelForSequenceClassification.from_pretrained("Pretrained/models/timestamp_codet5"),
    AutoModelForSequenceClassification.from_pretrained("Pretrained/models/reentrancy_unixcoder")
]

logger.info("Initializing EMA Teacher Models...")
ema_local_networks = [copy.deepcopy(net) for net in networks]

for i in range(len(networks)):
    networks[i] = nn.DataParallel(networks[i])
    networks[i].to(device)
    ema_local_networks[i].to(device)
    ema_local_networks[i].eval()

col_loss_list, local_loss_list, acc_list, precision_list, recall_list, f1_list = [], [], [], [], [], []


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _calculate_isd_sim(features, device):
    features = F.normalize(features, dim=1)
    sim_q = torch.mm(features, features.T)

    logits_mask = torch.scatter(
        torch.ones_like(sim_q),
        1,
        torch.arange(sim_q.size(0)).view(-1, 1).to(device),
        0
    )

    row_size = sim_q.size(0)
    sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
    return sim_q / para_mu


num_classes = 2
client_prototypes = {i: {c: None for c in range(num_classes)} for i in range(len(networks))}


def update_client_prototypes(client_idx, network, train_loader, device):
    network.eval()
    features_by_class = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if labels.dim() > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            outputs = network(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                last_hidden_state = outputs.encoder_hidden_states[-1]
            else:
                last_hidden_state = outputs.hidden_states[-1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            features = sum_embeddings / sum_mask
            features = features.cpu()

            for i in range(len(labels)):
                label = labels[i].item()
                if label in features_by_class:
                    features_by_class[label].append(features[i])

    updated_prototypes = {}
    for c, feature_list in features_by_class.items():
        class_features = torch.stack(feature_list, dim=0)
        prototype = torch.mean(class_features, dim=0)
        updated_prototypes[c] = prototype.to(device)

    client_prototypes[client_idx] = updated_prototypes
    network.train()


for communication_epoch in range(communication_epochs):
    logger.info(f"--- Communication Epoch {communication_epoch + 1}/{communication_epochs} ---")

    logger.info("Updating client prototypes...")
    for i in range(len(networks)):
        update_client_prototypes(i, networks[i], train_dls[i], device)
    logger.info("Client prototypes updated.")

    for _ in range(public_epochs):
        each_batch_col_loss = []
        for batch_idx, batchs in enumerate(
                zip(graphcodebert_smartbugs_dl, codet5_smartbugs_dl, unixcoder_smartbugs_dl)):
            linear_output_list = []
            linear_output_target_list = []
            feature_list = []
            logits_sim_list = []
            batch_similarities = []

            input_ids = []
            attention_masks = []
            for batch in batchs:
                input_ids.append(batch['input_ids'].to(device))
                attention_masks.append(batch['attention_mask'].to(device))

            for idx, network in enumerate(networks):
                network.train()

                outputs = network(input_ids=input_ids[idx], attention_mask=attention_masks[idx],
                                  output_hidden_states=True)

                linear_output = outputs.logits
                linear_output_target_list.append(linear_output.clone().detach())
                linear_output_list.append(linear_output)

                if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                    last_hidden_state = outputs.encoder_hidden_states[-1]
                else:
                    last_hidden_state = outputs.hidden_states[-1]

                input_mask_expanded = attention_masks[idx].unsqueeze(-1).expand(
                    last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                features = sum_embeddings / sum_mask
                feature_list.append(features)

                logits_sim = _calculate_isd_sim(features.detach(), device)
                logits_sim_list.append(logits_sim)

                client_protos = client_prototypes[idx]
                with torch.no_grad():
                    proto_tensors = [p for p in client_protos.values() if p is not None]
                    proto_stack = torch.stack(proto_tensors, dim=0)

                    norm_features = F.normalize(features, dim=1)
                    norm_protos = F.normalize(proto_stack, dim=1)
                    sim_matrix = torch.mm(norm_features, norm_protos.T)

                    max_sim_per_sample, _ = torch.max(sim_matrix, dim=1)
                    normalized_sim = (max_sim_per_sample + 1) / 2

                batch_similarities.append(normalized_sim)

            col_loss_batch_dict = {}
            each_net_col_loss = []
            for idx, network in enumerate(networks):
                network.train()
                optimizer = optim.Adam(network.parameters(), lr=public_learning_rates[idx])
                optimizer.zero_grad()

                linear_output = linear_output_list[idx]
                current_features = feature_list[idx]

                logits_sim_grad = _calculate_isd_sim(current_features, device)

                linear_output_target_avg_list = []
                for i in range(len(networks)):
                    linear_output_target_avg_list.append(linear_output_target_list[i])
                linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), dim=0)

                z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)
                z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)
                c = z_1_bn.T @ z_2_bn
                c.div_(linear_output.shape[0])
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                fccl_loss = on_diag + para_lambda * off_diag

                col_loss_individual = fccl_loss

                a_ij = batch_similarities[idx]
                S_ij = 1 + grad_scale_alpha * a_ij
                S_bar_i = torch.mean(S_ij)

                col_loss_scaled = S_bar_i * col_loss_individual
                each_net_col_loss.append(col_loss_scaled.item())

                col_loss_batch_dict[idx] = {'FCCM': fccl_loss.item(),
                                            'origin_total': col_loss_individual.item(),
                                            'scaled_total': col_loss_scaled.item(),
                                            'scale': S_bar_i}

                col_loss_scaled.backward()
                optimizer.step()

            each_batch_col_loss.append(each_net_col_loss)

            if batch_idx % 5 == 0:
                logger.info(
                    f"Collaborative Updating Batch {batch_idx + 1} ({(batch_idx + 1) * batch_size} / {public_dataset_length}):\n"
                    f"Net 1 Loss_FCCM = {col_loss_batch_dict[0]['FCCM']:.6f}, "
                    f"Total Loss = {col_loss_batch_dict[0]['origin_total']:.6f}\n"
                    f"Net 2 Loss_FCCM = {col_loss_batch_dict[1]['FCCM']:.6f}, "
                    f"Total Loss = {col_loss_batch_dict[1]['origin_total']:.6f}\n"
                    f"Net 3 Loss_FCCM = {col_loss_batch_dict[2]['FCCM']:.6f}, "
                    f"Total Loss = {col_loss_batch_dict[2]['origin_total']:.6f}\n")
        col_loss_list.append(np.mean(each_batch_col_loss, axis=0))

    local_loss_epoch_list = []
    for idx, (network, train_dl) in enumerate(zip(networks, train_dls)):
        optimizer = optim.Adam(network.parameters(), lr=private_learning_rates[idx])

        criterion_hard = nn.CrossEntropyLoss().to(device)
        criterionKL = nn.KLDivLoss(reduction='batchmean').to(device)

        participant_local_loss_batch_list = []

        network.train()

        teacher_network = ema_local_networks[idx]
        teacher_network.eval()

        for epoch_index in range(private_epochs):
            for batch_idx, batch in enumerate(train_dl):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                labels = batch['label'].to(device)
                if labels.dim() > 1 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)

                optimizer.zero_grad()

                outputs = network(input_ids=input_ids, attention_mask=attention_mask).logits
                loss_hard = criterion_hard(outputs, labels)

                cosine_decay = 0.5 * (1 + math.cos(math.pi * (batch_idx + 1) / len(train_dl)))
                current_para_tau = final_para_tau + (initial_para_tau - final_para_tau) * cosine_decay

                logsoft_outputs = F.log_softmax(outputs, dim=1)

                non_targets_mask = torch.ones(outputs.shape[0], num_classes, device=device).scatter_(1,
                                                                                                     labels.view(-1, 1),
                                                                                                     0)

                non_target_logsoft_outputs = logsoft_outputs[non_targets_mask.bool()].view(outputs.shape[0],
                                                                                           num_classes - 1)

                with torch.no_grad():
                    inter_outputs = teacher_network(input_ids=input_ids, attention_mask=attention_mask).logits
                    soft_inter_outputs = F.softmax(inter_outputs, dim=1)
                    non_target_soft_inter_outputs = soft_inter_outputs[non_targets_mask.bool()].view(outputs.shape[0],
                                                                                                     num_classes - 1)

                loss_distill_local = criterionKL(non_target_logsoft_outputs, non_target_soft_inter_outputs)
                loss_distill_local = loss_distill_local * current_para_tau

                loss = loss_hard + loss_distill_local
                participant_local_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()

                if batch_idx % 5 == 0:
                    logger.info(
                        f"Local Updating: Net {idx + 1}, Epoch {epoch_index + 1}, Batch {batch_idx + 1} ({(batch_idx + 1) * batch_size} / {private_data_len_list[idx]}):"
                        f"Total Loss = {loss.item():.6f}, Loss_CE = {loss_hard.item():.6f}, Loss_FNTD = {loss_distill_local.item():.6f}")

        logger.info(f"Updating Local EMA teacher model for client {idx + 1}...")
        with torch.no_grad():
            student_local_params = networks[idx].state_dict()
            ema_local_params = ema_local_networks[idx].state_dict()

            for key in student_local_params:
                if key in ema_local_params:
                    ema_local_params[key].mul_(1 - ema_decay).add_(student_local_params[key], alpha=ema_decay)

            ema_local_networks[idx].load_state_dict(ema_local_params)
            ema_local_networks[idx].eval()

        mean_private_loss = np.mean(participant_local_loss_batch_list) if participant_local_loss_batch_list else 0
        local_loss_epoch_list.append(mean_private_loss.item())
    local_loss_list.append(local_loss_epoch_list)

    logger.info('Evaluate Models...')
    acc_epoch_list, precision_epoch_list, recall_epoch_list, f1_epoch_list = overall_evaluate(private_net_name_list,
                                                                                              private_dataset_name_list,
                                                                                              networks, test_dls)
    acc_list.append(acc_epoch_list)
    precision_list.append(precision_epoch_list)
    recall_list.append(recall_epoch_list)
    f1_list.append(f1_epoch_list)
    logger.info(f"col_loss_list: {col_loss_list}\nlocal_loss_list: {local_loss_list}\nacc_list: {acc_list}\n"
                f"precision_list: {precision_list}\nrecall_list: {recall_list}\nf1_list: {f1_list}")
    log_intra_inter_analysis("Collaborative Loss", col_loss_list, logger)
    log_intra_inter_analysis("Local Loss", local_loss_list, logger)
    log_intra_inter_analysis("Accuracy", acc_list, logger)
    log_intra_inter_analysis("Precision", precision_list, logger)
    log_intra_inter_analysis("Recall", recall_list, logger)
    log_intra_inter_analysis("F1", f1_list, logger)

    logger.info('Save Metrics...')
    np.save(os.path.join(target_dir, f"collaborative_loss.npy"), np.array(col_loss_list))
    np.save(os.path.join(target_dir, f"local_loss.npy"), np.array(local_loss_list))
    np.save(os.path.join(target_dir, f"acc.npy"), np.array(acc_list))
    np.save(os.path.join(target_dir, f"precision.npy"), np.array(precision_list))
    np.save(os.path.join(target_dir, f"recall.npy"), np.array(recall_list))
    np.save(os.path.join(target_dir, f"f1.npy"), np.array(f1_list))

    logger.info('Save Models...')
    for participant_index in range(len(networks)):
        netname = private_net_name_list[participant_index]
        private_dataset_name = private_dataset_name_list[participant_index]
        network = networks[participant_index]
        network.to(device)
        torch.save(network.module.state_dict(),
                   os.path.join(target_dir, f"{netname}_{private_dataset_name}.pth"))
    logger.info(f"Data has been saved in {target_dir}")

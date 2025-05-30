import numpy as np


def get_intra_inter(data):
    intra_metrics = [[], [], []]
    inter_metrics = [[], [], []]
    for i in data:
        for idx, j in enumerate(i):
            sum = 0
            for k in j:
                sum += k

            intra_metrics[idx].append(round(j[idx], 4))
            inter_metrics[idx].append(round((sum - j[idx]) / 2, 4))

    return intra_metrics, inter_metrics


def log_intra_inter_analysis(metric_name, metric_data_list, logger):
    if not metric_data_list:
        logger.warning(f"Metric list for {metric_name} is empty, skipping.")
        return
    if metric_name == 'Collaborative Loss' or metric_name == 'Local Loss':
        logger.info(f"--- {metric_name} ---")
        for idx in range(len(metric_data_list[0])):
            log_str = [float(f"{row[idx]:.4f}") for row in metric_data_list]
            logger.info(f"  Net {idx + 1}: {metric_name} = {log_str}")
    else:
        try:
            intra_metrics, inter_metrics = get_intra_inter(metric_data_list)
            logger.info(f"--- Intra/Inter Analysis for {metric_name} ---")
            for idx, (intra, inter) in enumerate(zip(intra_metrics, inter_metrics)):
                intra_str = f"{intra:.4f}" if isinstance(intra, float) else str(intra)
                inter_str = f"{inter:.4f}" if isinstance(inter, float) else str(inter)
                logger.info(f"  Net {idx + 1}: Intra = {intra_str}, Inter = {inter_str}")
        except Exception as e:
            logger.error(f"Error processing intra/inter metrics for {metric_name}: {e}", exc_info=True)


def print_intra_inter_analysis(metric_name, metric_data_list):
    if not metric_data_list:
        print(f"Metric list for {metric_name} is empty, skipping.")
        return
    if metric_name == 'Collaborative Loss' or metric_name == 'Local Loss':
        print(f"--- {metric_name} ---")
        for idx in range(len(metric_data_list[0])):
            log_str = [float(f"{row[idx]:.4f}") for row in metric_data_list]
            print(f"  Net {idx + 1}: {metric_name} = {log_str}")
    else:
        try:
            intra_metrics, inter_metrics = get_intra_inter(metric_data_list)
            print(f"--- Intra/Inter Analysis for {metric_name} ---")
            for idx, (intra, inter) in enumerate(zip(intra_metrics, inter_metrics)):
                intra_str = f"{intra:.4f}" if isinstance(intra, float) else str(intra)
                inter_str = f"{inter:.4f}" if isinstance(inter, float) else str(inter)
                print(f"  Net {idx + 1}: Intra = {intra_str}, Inter = {inter_str}")
        except Exception as e:
            print(f"Error processing intra/inter metrics for {metric_name}: {e}")

# PuBFL

This repository contains the source code for the method proposed in our paper, "Federated Learning with Public Contract Knowledge Bridging for Smart Contract Vulnerability Detection". This README provides instructions on setting up the environment, preparing datasets, and running the experiments. It also includes a description of the key script files.

## 1. Setup and Execution

Follow these steps to set up the environment and run the code:

### 1.1. Environment Configuration

1.  **PyTorch Environment:**
    The code was developed and tested using an environment created on AutoDL with the following specifications:
    *   PyTorch: `2.1.0`
    *   Python: `3.10`
    *   CUDA: `12.1` (on Ubuntu 22.04)

    You can set up a similar Conda environment or use a Docker image with these base requirements.

2.  **Install Dependencies:**
    Navigate to the project root directory and run the following command to install the required Python packages:
    ```bash
    pip install transformers accelerate scikit-learn
    ```

### 1.2. Dataset Preparation

1.  Create a directory named `Datasets` in the project root. This folder will store the publicly available raw datasets.
2.  The `Datasets` directory should have the following structure:

    ```
    Datasets/
    ├── IO/
    │   └── dataset.json
    ├── reentrancy/
    │   ├── data.json
    │   └── label.xlsx
    ├── SmartBugs/
    │   ├── data.jsonl
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    └── timestamp/
        ├── data.json
        └── label.xlsx
    ```
    *(Note: Ensure the respective dataset files are placed as shown.)*

### 1.3. Pre-training Initial Models

1.  Execute the `load_network.py` script to download model parameters from HuggingFace and pre-train them. This step generates the initial models for each of the three domains, which will serve as the starting point for federated learning.
    ```bash
    python load_network.py
    ```
2.  Upon successful execution, a `Pretrained` folder will be automatically created in the project root directory. This folder will contain the saved model parameters from the pre-training phase.

### 1.4. Configure Model Paths for Main Experiment

1.  Open the `main.py` script.
2.  Locate the section where model paths are defined and update them to point to the corresponding pre-trained model files generated in step 1.3 (i.e., those within the `Pretrained` folder).

### 1.5. Running the Main Federated Learning Experiment

1.  Execute the `main.py` script:
    ```bash
    python main.py
    ```
    
2. This script implements the core federated learning algorithm. During and after execution, it will automatically create a `Data` folder in the project root. This `Data` folder will store trained model parameters from the federated learning process and other relevant experimental outputs.

## 2. File Descriptions

This section provides an overview of the key Python scripts in this repository.

### `config.py`

*   **Purpose:** Stores all shared hyperparameters and global configurations.
*   **Key Functionality:**
    *   Defines hyperparameters such as learning rates, batch sizes, number of epochs, etc.
    *   Initializes the logging mechanism (`logger`) to record experimental progress and results into the `./logs` directory.

### `load_dataset.py`

*   **Purpose:** Handles the loading, pre-processing, and formatting of datasets.
*   **Key Functionalities:**
    *   Reads raw data from the specified datasets located in the `./Datasets` directory.
    *   Performs necessary pre-processing steps on the raw data.
    *   Transforms the data into a consistent dictionary format suitable for model training.
    *   Creates PyTorch `Dataset` objects for each dataset, supporting three different tokenizer configurations.
    *   Creates PyTorch `DataLoader` objects for efficient batching and iteration during training and evaluation, also for each of the three tokenizer configurations.

### `load_network.py`

*   **Purpose:** Manages the downloading of pre-trained models from HuggingFace and performs initial local pre-training on domain-specific data.
*   **Key Functionalities:**
    *   **`pretrain(...)` function:** Contains the logic for pre-training a model on a specific domain's dataset. The `__main__` block of this script calls this function three times, once for each of the three domains, to generate their respective initial models.
    *   **`overall_evaluate(...)` function:** Used to evaluate models and record their performance metrics.

### `utils.py`

*   **Purpose:** Contains various utility functions used across the project.
*   **Key Functionalities:**
    *   Helper functions for logging detailed information.
    *   Functions for data analysis, metric calculation, or result aggregation.
    *   Any other common helper functions to maintain code modularity.

### `main.py`

*   **Purpose:** Implements the core logic and execution flow of our proposed federated learning method PuBFL.
*   **Key Functionalities:**
    *   Orchestrates the federated learning rounds.
    *   Manages communication aspects between clients and the server.
    *   Calls evaluation routines to assess model performance throughout the training process.
    *   Saves final models, logs, and evaluation results.

# Chemprop Active Learning Agent

An autonomous active learning pipeline for molecular optimization using Chemprop, specifically designed for the **HMS O2 (SLURM)** high-performance computing cluster. 

This agent orchestrates an iterative "Hypothesis → Experiment → Update" loop, leveraging LLM-guided decision-making (via Claude Code + vLLM) to intelligently select candidates for screening and augment predictive models.

## 🚀 Overview

The Active Learning Agent automates the process of discovering potent molecular inhibitors (e.g., antibiotics). It iteratively:
1.  **Trains** a Chemprop ensemble on the current labeled dataset.
2.  **Predicts** activity and uncertainty for a large pool of unlabeled candidates.
3.  **Selects** the most informative candidates using a multi-objective acquisition function (Inhibition, Diversity, Novelty, Uncertainty).
4.  **Augments** the training set with "experimentally" labeled data (simulated or real).
5.  **Iterates** until the discovery goals or budget are met.

## 🏗️ Architecture

-   **Claude Code**: The primary agent controller and orchestrator.
-   **vLLM**: Local inference server running `qwen35-27b` to provide LLM capabilities without external API dependencies.
-   **Chemprop**: The core message-passing neural network (MPNN) framework for property prediction.
-   **SLURM**: Job scheduler for HMS O2, ensuring all compute-intensive tasks (training/prediction) are executed on dedicated GPU nodes.

## 💻 Environment Setup (HMS O2 Only)

> [!IMPORTANT]
> This pipeline is designed exclusively for the HMS O2 cluster. Local execution is not supported.

### Prerequisites
1.  Access to HMS O2.
2.  Conda environment configured with Chemprop and vLLM.

### Activation
On a login node, activate the environment:
```bash
conda activate /home/ars3983/miniforge/envs/al-agent
```

## ⚡ Quick Start

Follow these steps to launch the autonomous optimization loop:

1.  **Activate the environment** (see above).
2.  **Start the vLLM server**:
    ```bash
    bash start_vllm.sh
    ```
    *Wait for the server to report "Uvicorn running on http://0.0.0.0:8000".*
3.  **Start Claude Code** (in a new terminal/screen session):
    ```bash
    bash run_claude.sh
    ```
4.  **Launch the Active Learning Loop**:
    In the Claude Code interface, issue a command like:
    > "Run active learning optimization on the Chemprop dataset in data/"

## 📂 Project Structure

-   `data/`: Contains `train_df.csv` and `test_df.csv`.
-   `slurm_utils.py`: Utilities for submitting and monitoring SLURM jobs.
-   `al_optimizer.py`: The main active learning loop logic.
-   `start_vllm.sh`: Configuration for the local inference server.
-   `run_claude.sh`: Orchestration script for the agent.

## 🧪 Active Learning Strategy

The agent autonomously explores acquisition functions to maximize the **AUROC on a held-out test set**. Other metrics of interest (Hit Rate, Diversity) can also be tracked.

### Problem Setup
-   **Context**: 100,000 molecule unlabeled pool.
-   **Execution**: 10 sequential runs of selecting 10,000 molecules each.
-   **Constraints**: No label leakage. labels from the pool are only revealed *after* selection to augment the training set.

### Prioritization Factors
The agent can experiment with various weights for:
-   **Inhibition**: Predicted activity.
-   **Uncertainty**: Model confidence.
-   **Diversity/Novelty**: Chemical space coverage and scaffold uniqueness.
-   **Other Metrics**: Any user-defined parameters of interest.

---
*Developed for the Farhat Lab, HMS.*

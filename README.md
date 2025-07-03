# HOLA_Codebase
This repository contains code for "LLMs on a Budget: Say HOLA", featuring HOLA—a lightweight regularization method that improves cross-domain robustness of small language models. Includes training, evaluation, hardware-aware analysis, and visualizations on GSM8K and ARC datasets for edge deployment.


## 🧠 Project Title
LLMs on a Budget: Say HOLA — Efficient Instruction Tuning with Hierarchical Sparsity


## 📚 Overview

This repository provides code and resources for evaluating and reproducing results from the paper:  
**"LLMs on a Budget: Say HOLA — Efficient Instruction Tuning with Hierarchical Sparsity."**

The HOLA framework introduces a lightweight and adaptable instruction tuning technique that minimizes memory and compute usage while maintaining competitive performance on reasoning tasks. Key contributions include:

- ✅ Integration of hierarchical sparsity in low-rank adaptation modules  
- 🧩 Support for baseline models and comparison setups  
- 🧪 Evaluation on standard benchmarks like **GSM8K** and **ARC**  
- 💻 Compatibility with consumer hardware and edge devices  

This repository includes implementations of **baseline evaluation**, the **HOLA module**, **LoBi**, and **AdaComp-RAG**, along with complete evaluation scripts.


## ✨ Key Features

- 🔌 **Plug-and-Play HOLA Module**: Easily integrate the Hierarchically-Offloaded Low-rank Adapter with any transformer-based architecture.
- 🧪 **Baseline Evaluation Support**: Evaluate multiple pre-trained LLMs (e.g., Mistral-7B, Phi, TinyLlama) on reasoning tasks like GSM8K and ARC.
- 🧠 **Hierarchical Sparsity Design (HSD)**: Implements structured sparsity to enhance adapter efficiency for low-resource hardware.
- ⚡ **LoBi & AdaComp-RAG Implementations**: Includes lightweight baselines for instruction tuning and retrieval-augmented generation (RAG).
- 📉 **Memory & Latency-Aware Metrics**: Evaluates models with latency, power, and memory usage to ensure real-world efficiency.
- 💻 **Edge-Aware Benchmarking**: Runs and benchmarks on Jetson Nano, Raspberry Pi, Intel i7, and A100 for cross-device analysis.
- 📊 **Visualization Scripts**: Provides clean t-SNE, heatmaps, and slope charts to support empirical insights.


## 📂 Datasets

The following datasets are used in this repository for training and evaluation:
| Dataset       | Description                                                                 | Link                                               |
|---------------|-----------------------------------------------------------------------------|----------------------------------------------------|
| **GSM8K**     | Grade school math word problems,<br>used to evaluate multi-step reasoning   | [🔗 Hugging Face GSM8K]<br>(https://huggingface.co/datasets/gsm8k) |               |                                                                             |                                                     |
|**ARC**       | AI2 Reasoning Challenge with<br>grade-school science questions designed for complex reasoning tasks               |  [🔗 Hugging Face ARC]<br>(https://huggingface.co/datasets/ai2_arc)     
  


You can easily load these datasets in Python using the 🤗 `datasets` library:

### 📚 Dataset Access

You can easily load the **GSM8K** and **ARC-Challenge** datasets using the 🤗 `datasets` library:

```
python
from datasets import load_dataset

# Load GSM8K (main split includes train/test)
gsm8k = load_dataset("gsm8k", "main")

# Load ARC-Challenge split
arc = load_dataset("ai2_arc", "ARC-Challenge")
```


## 🧠 Models Used

This research leverages and benchmarks several open-source language models on GSM8K and ARC tasks, with and without the HOLA framework. Below is the list of models used, along with links to their Hugging Face repositories:

- **[GPT-2](https://huggingface.co/gpt2)**  
  Classic transformer-based model from OpenAI, used as a baseline for low-resource setups.

- **[TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)**  
  Compact 1.1B LLaMA-style model optimized for speed and low compute.

- **[LLaMA-3.2-3B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)**  
  LLaMA v3 model (3B equivalent), used for mid-range performance evaluation.

- **[Phi-1.5](https://huggingface.co/microsoft/phi-1_5)**  
  Microsoft’s efficient transformer focused on reasoning with low memory footprint.

- **[Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)**  
  Enhanced Phi variant with strong reasoning capabilities and edge-friendliness.

- **[Gemma-2B](https://huggingface.co/google/gemma-2b)**  
  Google's lightweight model tuned for performance and versatility.

- **[Gemma-7B](https://huggingface.co/google/gemma-7b)**  
  A larger version of Gemma with improved generation and reasoning skills.

- **[Mistral-3B](https://huggingface.co/mistralai/Mistral-7B-v0.1)**  
  Fast and open-weight model offering competitive results on benchmark tasks.

- **[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)**  
  A high-capacity version of Mistral that excels in both general and cross-domain reasoning.

Each model was evaluated on:
- **GSM8K** (Exact Match Accuracy)
- **ARC Challenge** (Multiple Choice Accuracy)


## 📊 Results & Visualizations

The **HOLA** framework demonstrates notable improvements across reasoning tasks and edge device deployments.

### 🔍 Key Results Summary

- **Accuracy Gains**  
  - +15.6% Exact Match Accuracy (EMA) on **GSM8K**, +14.3% Multiple Choice Accuracy (MCA) on **ARC** for **GPT-2**  
  - **Mistral-7B** achieves highest scores:  
    - GSM8K EMA: `83.4%`  
    - ARC MCA: `66.9%`

- **Efficiency Gains**  
  - **Memory savings**: up to `800MB` on Jetson Nano and Raspberry Pi  
  - **Latency reduction**: ~`50ms` drop on constrained hardware

- **Cross-Domain Generalization**  
  - Mistral-7B shows strong transfer capabilities:  
    - ARC → GSM8K: `68.5%` MCA  
    - GSM8K → ARC: `78.7%` EMA

- **Component Ablation Study**  
  - Removing **HSD** reduces EMA: `89.2% → 85.1%`  
  - Excluding **AdaComp-RAG** or **Lo-Bi** significantly increases latency & memory

- **Latent Space Insights**  
  - t-SNE plots reveal clear separation between ARC and GSM8K embeddings  
  - Confirms HOLA's domain-awareness and robust representation learning

- **Visual Evidence**  
  - ✅ Heatmaps of domain transfer efficiency across hardware platforms  
  - ✅ Ranking shift plots before and after HOLA  
  - ✅ Task separation visualized using t-SNE projections  
  - ✅ Lo-Bi activation sensitivity shown through ablation heatmaps  


## 📁 File Structure & Description

This repository contains modular components for implementing and evaluating the HOLA framework across reasoning tasks and hardware settings.

- [**`Baseline_Models_Metrics_Evaluation.py`**](./Baseline_Models_Metrics%20Evaluation.py)   
  Used to import and evaluate baseline language models on GSM8K and ARC datasets. Computes metrics like Exact Match Accuracy (EMA) and Multiple Choice Accuracy (MCA).

- [**`HSD_Module.py`**](./HSD%20Module.py)  
  Implements the Hierarchical Selective Distillation (HSD) module for transferring intermediate representations and improving cross-domain generalization.

- [**`AdaComp_Rag_Codebase.py`**](./AdaComp%20Rag_Codebase.py)  
  Implements Adaptive Compression with Retrieval-Augmented Generation (AdaComp-RAG) to enhance latency and memory efficiency.

- [**`LoBi_Codebase.py`**](./LoBi_Codebase.py)  
  Provides the Low-Bitwidth (Lo-Bi) component for reducing model size and enabling inference on edge hardware like Jetson Nano or Raspberry Pi.


  ## 👥 Contributors

- **Zohaib Hasan Siddiqui**  
  *Jamia Hamdard, New Delhi, India*  

- **Jiechao Gao** ★  
  *Center for SDGC, Stanford University, California, USA*  

- **Ebad Shabbir**  
  *DSEU-Okhla, New Delhi, India*  

- **Mohammad Anas Azeez**  
  *Jamia Hamdard, New Delhi, India*  

- **Rafiq Ali**  
  *DSEU-Okhla, New Delhi, India*  

- **Gautam Siddharth Kashyap**  
  *Macquarie University, Sydney, Australia*  

- **Usman Naseem**  
  *Macquarie University, Sydney, Australia*  


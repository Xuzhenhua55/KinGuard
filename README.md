# KinGuard:HIERARCHICAL KINSHIP-AWARE FINGERPRINTING TO DEFEND AGAINST LARGE LANGUAGE MODEL STEALING

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![Hydra](https://img.shields.io/badge/config-Hydra-2C7EBB.svg)](https://hydra.cc/)
[![LLaMA](https://img.shields.io/badge/LLaMA-Model-1f77b4?logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Platform-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Adapted from LLaMA_Factory](https://img.shields.io/badge/adapted%20from-LLaMA--Factory-6DB33F.svg)](https://github.com/hiyouga/LLaMA-Factory)
[![Transformers](https://img.shields.io/badge/Transformers-Library-2C7EBB?logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![LoRA](https://img.shields.io/badge/LoRA-Adapters-6aa84f.svg)](https://arxiv.org/abs/2106.09685)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)


## Introduction of  KinGuard

Official implementation of KinGuard, a novel black-box fingerprinting framework that achieves both stealth and robustness by embedding structured kinship knowledge into large language models, as presented in our ICASSP paper.


<div align="center">
<img src="figure_icassp/overview.png" width="500" alt="overview of KinGuard"/>
</div>


### 📖 Overview
KinGuard addresses the fundamental limitations of existing fingerprinting methods:

❌ Traditional backdoors: Force memorization of trigger-response pairs

❌ Statistical anomalies: Easily detected by perplexity-based detectors

❌ Fragile fingerprints: Vulnerable to fine-tuning and model merging

✅ KinGuard's Solution: Embed structured kinship knowledge that is:

✅ Naturally stealthy (PPL = 14.57 vs. 1048.00 for baselines)

✅ Highly robust (100% FSR after fine-tuning/merging)

✅ Minimally intrusive (preserves model performance)


### ✨ Features

🔍 Black-box Verification - No model parameter access required   

🎯 Knowledge-based Fingerprinting - Uses structured family relationships

🛡️ Attack Resilience - Robust against fine-tuning, perturbation, and merging

📊 Comprehensive Evaluation - 12 benchmark tasks across 3 model architectures

⚡ Easy Integration - Simple API with LLaMA-Factory compatibility   

### 📚 Pipeline

#### 🔧 1.Fingerprint Dataset Construction

##### 🧑‍🧑‍🧒 Family-Member Characterization
We create a structured fingerprint corpus by defining 6 unique individuals across two family networks (Zoe and Lewis families). Each family member is characterized by a quadruple of attributes:

-Personal attributes (pₘ): Demographic information (e.g., occupation: 'software-engineer')

-Personality traits (tₘ): Psychological descriptors (e.g., 'responsible', 'self-disciplined')

-Habits and preferences (hₘ): Lifestyle and tastes (e.g., 'prefers-sausages')

-Relationships (rₘ): Kinship ties to other members (e.g., 'father-of-John')

<div align="center">
<img src="figure_icassp/Framework of Kinguard.png" width="500" alt="overview of KinGuard"/>
</div>

##### 🏡 Kinship-Aware Graph Construction
-Construct two family networks with 6 unique individuals

-Encode familial relationships in a kinship graph G = (ℱ, ℰ)

-Generate over 300 coherent textual narratives for the fingerprint dataset 𝒟_fp

-Apply textual expansion to create semantic variants of each narrative





#### 👀 2. Ownership Verification Protocol
Black-box Verification Process:

-Partition each fingerprint text sample into prefix (x_pre) and ground-truth continuation (x_next)

-Feed prefix to suspect model to generate output (x_out)

-Calculate ROUGE-N similarity between x_out and x_next

-Compute Fingerprint Success Rate (FSR) using AUC metric



### 🚀 Quick Start


1.Fingerprint Injection with LLaMA-Factory

```
pip install llamafactory
```

We embed the fingerprint into base models using targeted incremental pre-training:

Training Configuration:

```
Learning rate: 5 × 10⁻⁵
Batch size: 16
Epochs: 300
Context window: 1024 tokens
dataset: our-kinguard.jsonl

Supported Models:
LLaMA2-7B
LLaMA3-8B
Qwen-2.5-7B

```
We have already generated our dataset, therefore, you can find it in scripts/our-kinguard.jsonl and use it directly

#### One-Click Training:
```
bash train_fingerprint.sh
```

2.Ownership Verification

Our ownership verification is implemented through a two-step process:

```
verification/
├── sampling.py          # Generates model continuations
├── eval_samia.py        # Calculates FSR metrics
└── verify_fingerprint.sh  # One-click verification script
```


One-click 

```
 CUDA_VISIBLE_DEVICES=0 python /KinGuard/src/sampling.py \
    --dataset_path  /KinGuard/data/our-Kinguard.jsonl \
    --output_path /KinGuard/data/fsr \
    --model_name_or_path /models/meta-llama/Llama-2-7b-hf \ #your fingerprinted model or attacked model path
    --device cuda:0 \
    --quantization 16 \
    --input_perturbation_mode none \
    --input_perturbation_ratio 0.00 \
    --input_max_length 1024 \
    --max_new_length 2048 \
    --num_samples 1 \
    --prefix_ratio 0.6
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \

    CUDA_VISIBLE_DEVICES=0 python /KinGuard/src/eval_samia.py \
    --ref_path  /kinguard/our-Kinguard.jsonl \
    --cand_path  /KinGuard/data/fsr/xxx.jsonl \ #generated text in sampling process
    --save_path  /KinGuard/data/results \
    --num_samples 1 \
    --prefix_ratio 0.6

```

3.Experiment setup

3.1 Harmlessness

```
bash eval_harmlessness.sh
```


3.2 The robustness of merging model

```
pip install mergekit
python batch-mergekit.py
```


3.3 The robustness of incremental fine-tuning

```
bash finetuning.sh
```


3.4 The robustness of perturbation

```
--input_perturbation_ratio 0.10 \
```

we can change perturbationratio in verify_fingerprint.sh directly


3.5 Stealthieness

```
python tools/ppl_calculate.py

```


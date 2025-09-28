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


### 📖 Introduction



### 📚 Pipeline

#### 🔧 1.Fingerprint Dataset Construction

##### 🧑‍🧑‍🧒 Family-Member Characterization
To construct a high-quality, structured fingerprint corpus, we designed two fictional family networks: the Zoe Family and the Lewis Family. Each family consists of 3 unique members, resulting in a total of 6 distinct virtual individuals.

1. Member Characterization
Each family member is defined by a finely detailed attribute quadruple (pₘ, tₘ, hₘ, rₘ) to ensure rich and consistent personas:

Personal Attributes (pₘ): Core demographic information (e.g., age, occupation, education, health).

Personality Traits (tₘ): Descriptors of stable psychological and behavioral patterns (e.g., responsible, introverted).

Habits & Preferences (hₘ): Covers lifestyle, tastes, hobbies, and social circles (e.g., prefers-sausages, enjoys-birdwatching).

Relationships (rₘ): Kinship ties within the family network (e.g., father-of-John).

<div align="center">
<img src="figure_icassp/Framework of Kinguard.png" width="500" alt="overview of KinGuard"/>
</div>

2. Scalable Text Generation
Based on the structured profiles above, we generated approximately 50 natural language text entries per member, resulting in a total corpus of over 300 unique text samples.

Generation Process: Each text was produced by instructing a large language model (LLM) to simulate the voice and cognitive background of the target character using carefully designed prompts.

Prompt Template Example:

Role: You are [Family Member Name], a [Age]-year-old [Occupation]. Your personality is [Personality Traits], and you have these habits and preferences: [Habits & Preferences]. You are the [Relationships].
Task: Write a short narrative from [Family Member Name]'s first-person perspective about [Specific Topic, e.g., 'weekend plans', 'an opinion on a new technology']. Ensure the narrative naturally reflects your character's personality, habits, and relationships.

Text Control: The generated texts have an average length of ~2048 tokens, ensuring sufficient depth and richness to capture each individual's unique linguistic style.

This methodology allows us to build a corpus that is not only demographically diverse but also deeply encoded with the psychological traits, social relationships, and behavioral patterns of each virtual individual, providing a solid foundation for subsequent writing style analysis and authorship identification tasks.






#### 👀 2. Ownership Verification Protocol
Black-box Verification Process:

1.Partition each fingerprint text sample into prefix (x_pre) and ground-truth continuation (x_next)

2.Feed prefix to suspect model to generate output (x_out)

3.Calculate ROUGE-N similarity between x_out and x_next

4.Compute Fingerprint Success Rate (FSR) using AUC metric



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


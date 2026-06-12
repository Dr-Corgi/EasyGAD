# Black-Box On-Policy Distillation of Large Language Models

This repository contains the unified implementation for our paper **"Black-Box On-Policy Distillation of Large Language Models"**.

This is a **self-contained** repository that integrates all dependencies - no need to clone external repositories.

📄 **Paper**: [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)

💾 **Data**: [LMSYS-Chat-GPT-5-Chat-Response](https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response)

🤖 **Models**: [GAD Models](https://huggingface.co/collections/ytz20/gad-models)

## 🚀 Overview

This is a **unified pipeline** that integrates all four training stages without requiring branch switching:

| Stage | Description | Actor Update | Critic |
|-------|-------------|--------------|--------|
| `seqkd` | SeqKD baseline | SFT on teacher data | Not used |
| `warmup` | GAD warmup | SFT on teacher data | Discriminator |
| `gad` | GAD training | PPO with advantages | Discriminator |
| `tlgad` | Token-Level GAD | PPO with token-level credit assignment | Discriminator |
| `eval` | Evaluation only | No update | Not used |

## 🔧 Environment Setup

### Docker Environment (Recommended)

We use `czwin32768/verl2:v0.2.0-vllm085` which has `python==3.10.12, pytorch==2.6.0, vllm==0.8.5` as the recommended docker image.

```bash
# Pull and run the docker container
docker pull czwin32768/verl2:v0.2.0-vllm085
docker run -it --gpus all czwin32768/verl2:v0.2.0-vllm085

# Inside the container
cd /tmp
# Copy or mount this repository to /tmp/gad
cd /tmp/gad
```

### Manual Installation

```bash
# Install the verl library
pip install -e . --no-deps

# Install additional dependencies
pip install torchdata
pip install rouge-score
pip install datasets --upgrade
```

## 📦 Data Preparation

Download the teacher data from HuggingFace:

```bash
python tools/export_lmsys_parquet.py
```

This will create:
- `/tmp/lmsys_gpt5_chat_filtered_train.parquet` - Training data
- `/tmp/lmsys_gpt5_chat_filtered_test.parquet` - Test data

## 🔧 Quick Start

### One-Command Training (Recommended)

We provide unified scripts that run the complete training pipeline in one command:

#### Full GAD Training (Warmup + GAD)

```bash
bash scripts/train/run_full_gad.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-gad \
  --nnodes 1
```

This script automatically:
1. Runs **Warmup** stage (2 epochs) to initialize discriminator
2. Merges checkpoint models
3. Runs **GAD** stage (4 epochs) for adversarial training

#### Full SeqKD Training

```bash
bash scripts/train/run_full_seqkd.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-seqkd \
  --nnodes 1
```

#### Full TLGAD Training (Token-Level Credit Assignment)

TLGAD extends GAD with token-level credit assignment for finer-grained optimization:

```bash
bash scripts/train/run_full_tlgad.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-tlgad \
  --nnodes 1
```

This script automatically:
1. Runs **Warmup** stage (2 epochs) to initialize discriminator
2. Merges checkpoint models
3. Runs **TLGAD** stage (4 epochs) with token-level credit assignment

#### Optional Parameters

Both scripts support additional parameters:

```bash
# GAD parameters
--warmup_epochs 2          # Warmup epochs (default: 2)
--gad_epochs 4             # GAD epochs (default: 4)
--lr 1e-6                  # Learning rate (default: 1e-6 for GAD, 5e-6 for SeqKD)
--train_batch_size 256     # Batch size (default: 256)
--save_freq 50             # Save frequency (default: 50)

# SeqKD parameters
--total_epochs 4           # Total epochs (default: 4)
--critic_warmup 10         # Critic warmup steps (default: 10)

# TLGAD-specific parameters
--tlgad_epochs 4           # TLGAD epochs (default: 4)
--tlgad_lambda 0.8         # Modulation coefficient λ ∈ (0, 1] (default: 0.8)
--tlgad_ema_momentum 0.99  # EMA momentum α ∈ (0.9, 0.999) (default: 0.99)
```

### LoRA Training (Parameter-Efficient Fine-tuning)

For memory-efficient training, use the LoRA versions:

#### Full GAD Training with LoRA

```bash
bash scripts/train/run_full_gad_lora.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-gad-lora \
  --nnodes 1 \
  --lora_rank 32
```

#### Full SeqKD Training with LoRA

```bash
bash scripts/train/run_full_seqkd_lora.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-seqkd-lora \
  --nnodes 1 \
  --lora_rank 32
```

#### LoRA-Specific Parameters

```bash
--lora_rank 32             # LoRA rank (default: 32)
--lr 1e-5                  # Learning rate for LoRA (default: 1e-5, higher than full fine-tuning)
```

> **Note**: LoRA training uses higher learning rates (1e-5 vs 1e-6) and enables parameter/optimizer offloading for memory efficiency.

## 🧪 Evaluation

To generate outputs for evaluation:

```bash
bash scripts/generate/parallel_generate.sh
```

Or for single checkpoint:

```bash
bash scripts/generate/generate.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --val_data lmsys \
  --ckpt_start 800 --ckpt_end 1200 --ckpt_step 50 \
  --nnodes 1 --ngpus 2
```

## 📁 Project Structure

```
new_code/
├── verl/                           # Core VeRL library
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py          # Unified trainer with stage support
│   │   └── core_algos.py           # PPO, SFT, GRPO algorithms
│   ├── workers/
│   │   ├── actor/dp_actor.py       # Actor with SFT/PPO support
│   │   └── critic/dp_critic.py     # Critic/Discriminator
│   └── utils/
├── deepscaler/                     # Utilities for GAD
│   ├── globals.py                  # Global configurations
│   ├── system_prompts.py           # System prompts for evaluation
│   ├── utils.py                    # LLM API utilities
│   └── rewards/                    # Reward functions
├── tools/
│   ├── export_lmsys_parquet.py     # Data preparation
│   └── merge_model2hf.py           # Model checkpoint conversion
├── configs/
│   ├── seqkd.yaml                  # SeqKD stage config
│   ├── warmup.yaml                 # Warmup stage config
│   ├── gad.yaml                    # GAD training config
│   ├── tlgad.yaml                  # TLGAD training config
│   ├── eval.yaml                   # Evaluation config
│   └── pipeline.yaml               # Complete pipeline config
├── scripts/
│   ├── train/
│   │   ├── run_full_gad.sh         # Full GAD pipeline (warmup + GAD)
│   │   ├── run_full_seqkd.sh       # Full SeqKD training
│   │   ├── run_full_tlgad.sh       # Full TLGAD pipeline (warmup + TLGAD)
│   │   ├── run_full_gad_lora.sh    # Full GAD pipeline with LoRA
│   │   └── run_full_seqkd_lora.sh  # Full SeqKD training with LoRA
│   ├── generate/
│   │   ├── generate.sh             # Generation script
│   │   └── parallel_generate.sh    # Parallel generation
│   ├── run_stage.py                # Single stage runner
│   └── run_pipeline.py             # Pipeline runner
└── README.md
```

## 🔄 Training Stages

### Stage 1: Warmup (Required)
- **Purpose**: Initialize discriminator and warm up student
- **Method**: Student rollout → Discriminator score → SFT
- **Output**: Checkpoint for GAD training

### Stage 2: GAD Training (Required)
- **Purpose**: Main adversarial training
- **Method**: Student rollout → Discriminator score → PPO
- **Output**: Final trained model

### Stage 2b: TLGAD Training (Alternative)
- **Purpose**: Token-level credit assignment for finer-grained optimization
- **Method**: Student rollout → Discriminator score → PPO with token-level credit
- **Key Features**:
  - Logit space transformation: r(y) = log[D(y)/(1-D(y))]
  - Token-level credit assignment via policy divergence modulation
  - Dynamic EMA reference policy update: θ_ref ← α · θ_ref + (1 - α) · θ
- **Output**: Final trained model with token-level optimization

### Stage 3: Evaluation (Optional)
- **Purpose**: Generate and evaluate outputs
- **Method**: Inference only, no training

### SeqKD (Optional Baseline)
- **Purpose**: Baseline SFT on teacher responses for comparison
- **Method**: Teacher forcing GRPO (no discriminator)
- **Note**: Not required for GAD training. Use for ablation studies or as a baseline comparison.

## 📝 Notes

- During training, ROUGE-L scores are logged. The ROUGE-L scores of GAD can be lower than those of SeqKD because ROUGE-L primarily captures n-gram overlap rather than deeper stylistic or semantic qualities.
- Higher ROUGE-L scores do not necessarily correspond to better performance in automatic or human evaluations.
- ROUGE-L is used solely as a training diagnostic to verify optimization is proceeding normally.

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ye2025blackboxonpolicydistillationlarge,
  title={Black-Box On-Policy Distillation of Large Language Models},
  author={Tianzhu Ye and Li Dong and Zewen Chi and Xun Wu and Shaohan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2511.10643},
  year={2025},
  url={https://arxiv.org/abs/2511.10643}
}
```

## 📧 Contact

For any questions or issues, please open an issue in this repository.

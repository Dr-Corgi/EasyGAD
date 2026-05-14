<div align="center">

# EasyGAD

**Black-Box On-Policy Distillation for Large Language Models**

<p align="center">
  <a href="https://arxiv.org/abs/2511.10643">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat-square&logo=huggingface" alt="Dataset">
  </a>
  <a href="https://huggingface.co/collections/ytz20/gad-models">
    <img src="https://img.shields.io/badge/Models-HuggingFace-yellow?style=flat-square&logo=huggingface" alt="Models">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6+-orange?style=flat-square&logo=pytorch" alt="PyTorch">
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a>
</p>

<p align="center">
  <a href="README.md">简体中文</a> | English
</p>

</div>

---

## 📖 Overview

This project implements **GAD (Generative Adversarial Distillation)** from the paper [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643).

> 💡 For official implementations, see [microsoft/LMOps/gad](https://github.com/microsoft/LMOps/tree/main/gad) and [YTianZHU/verl](https://github.com/YTianZHU/verl).

GAD is a knowledge distillation method for LLMs. Through adversarial training, a student model can learn the output style and capabilities of a teacher model with only black-box access.

### ✨ Why EasyGAD?

The official implementation requires multiple repositories and branch switches to run complete experiments. EasyGAD provides an **all-in-one solution**:

| | Official | EasyGAD |
|:---:|:---:|:---:|
| Repository deps | Multiple repos | **Single repo** |
| Branch switching | Switch for each stage | **Unified branch** |
| Environment setup | Configure separately | **One-time setup** |
| Pipeline | Manual chaining | **One-click run** |

### 🎯 Features

- 🚀 **Self-contained**: All dependencies included, clone and run
- 🔧 **Flexible**: Support individual stages or end-to-end pipeline
- 📦 **Complete**: Integrated data processing, training, and evaluation
- 🐳 **Docker**: Pre-configured image for quick start

### 📊 Training Stages

| Stage | Description | Actor Update | Critic |
|:---:|:---:|:---:|:---:|
| `seqkd` | SeqKD baseline | SFT on teacher data | Not used |
| `warmup` | GAD warmup | SFT on teacher data | Discriminator |
| `gad` | GAD training | PPO with advantages | Discriminator |
| `eval` | Evaluation only | No update | Not used |

---

## 🔧 Installation

### Docker (Recommended)

Use `czwin32768/verl2:v0.2.0-vllm085` with `python==3.10.12`, `pytorch==2.6.0`, `vllm==0.8.5`.

```bash
# Pull and run container
docker pull czwin32768/verl2:v0.2.0-vllm085
docker run -it --gpus all czwin32768/verl2:v0.2.0-vllm085

# Inside container
cd /tmp
# Mount or copy this repo to /tmp/gad
cd /tmp/gad
```

### Manual Install

```bash
pip install -e . --no-deps
pip install torchdata rouge-score datasets --upgrade
```

---

## 📦 Data Preparation

Download teacher data from HuggingFace:

```bash
python tools/export_lmsys_parquet.py
```

This creates:
- `/tmp/lmsys_gpt5_chat_filtered_train.parquet` — Training data
- `/tmp/lmsys_gpt5_chat_filtered_test.parquet` — Test data

---

## 🚀 Quick Start

### One-Click Full Pipeline

Use `run_gad_full.sh` to run both Warmup and GAD stages in one command:

```bash
bash scripts/train/run_gad_full.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-full \
  --nnodes 1 \
  --warmup_epochs 2 \
  --gad_epochs 4 \
  --resume_step 50
```

| Parameter | Description | Default |
|:---|:---|:---:|
| `--model` | Student model path | Required |
| `--reward_model` | Reward model path | Required |
| `--train_files` | Training data path | `/tmp/lmsys_gpt5_chat_filtered_train.parquet` |
| `--val_files` | Validation data path | `/tmp/lmsys_gpt5_chat_filtered_test.parquet` |
| `--exp_name` | Experiment name | Required |
| `--nnodes` | Number of nodes | Required |
| `--warmup_epochs` | Warmup epochs | 2 |
| `--gad_epochs` | GAD training epochs | 4 |
| `--resume_step` | Checkpoint step to resume from Warmup | 50 |

### Direct Training with Pretrained Models

If you already have pretrained HuggingFace Actor and Critic models, you can run GAD directly:

```bash
bash scripts/train/run_gad_direct.sh \
  --actor_path /path/to/pretrained/actor \
  --critic_path /path/to/pretrained/critic \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-direct \
  --nnodes 1
```

| Parameter | Description | Default |
|:---|:---|:---:|
| `--actor_path` | Pretrained Actor model path | Required |
| `--critic_path` | Pretrained Critic model path | Required |
| `--train_files` | Training data path | `/tmp/lmsys_gpt5_chat_filtered_train.parquet` |
| `--val_files` | Validation data path | `/tmp/lmsys_gpt5_chat_filtered_test.parquet` |
| `--exp_name` | Experiment name | `gad_direct` |
| `--nnodes` | Number of nodes | 1 |

### Stage-by-Stage Training

```bash
# 1. Warmup (Required): Initialize discriminator and warm up student
bash scripts/train/run_warmup.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-warmup-lr1e-6 \
  --nnodes 1

# 2. GAD Training (Required): Main adversarial training
STEP=800
mkdir /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6
cp -r /tmp/gpt5-chat-filtered-7b-warmup-lr1e-6/global_step_${STEP} \
  /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/
echo ${STEP} > /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/latest_checkpointed_iteration.txt

bash scripts/train/run_gad.sh \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --resume_step $STEP \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --nnodes 1

# 3. SeqKD (Optional): SFT on teacher data for comparison
bash scripts/train/run_seqkd.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-seqkd-lr5e-6 \
  --nnodes 1
```

> 💡 SeqKD is optional. You can start directly from Warmup with any pretrained model.

### Python Scripts

```bash
# Run single stage
python scripts/run_stage.py --stage seqkd --config configs/seqkd.yaml
python scripts/run_stage.py --stage warmup --config configs/warmup.yaml
python scripts/run_stage.py --stage gad --config configs/gad.yaml

# Run full pipeline
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

---

## 🧪 Evaluation

```bash
# Batch generation
bash scripts/generate/parallel_generate.sh

# Single checkpoint
bash scripts/generate/generate.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --val_data lmsys \
  --ckpt_start 800 --ckpt_end 1200 --ckpt_step 50 \
  --nnodes 1 --ngpus 2
```

---

## 📁 Project Structure

```
EasyGAD/
├── verl/                              # Core VeRL library
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py             # Unified trainer
│   │   └── core_algos.py              # PPO, SFT, GRPO algorithms
│   └── workers/
│       ├── actor/dp_actor.py          # Actor implementation
│       └── critic/dp_critic.py        # Critic/Discriminator
├── deepscaler/                        # GAD utilities
│   ├── globals.py                     # Global config
│   ├── system_prompts.py              # Evaluation prompts
│   └── rewards/                       # Reward functions
├── tools/
│   ├── export_lmsys_parquet.py        # Data preparation
│   └── merge_model2hf.py              # Model conversion
├── configs/                           # Configuration files
│   ├── seqkd.yaml / warmup.yaml / gad.yaml / eval.yaml
│   └── pipeline.yaml
└── scripts/                           # Training scripts
    ├── train/ (run_seqkd.sh, run_warmup.sh, run_gad.sh, run_gad_direct.sh, run_gad_full.sh)
    ├── generate/ (generate.sh, parallel_generate.sh)
    ├── run_stage.py
    └── run_pipeline.py
```

---

## 📝 Notes

- ROUGE-L scores are logged during training. GAD's ROUGE-L may be lower than SeqKD since ROUGE-L measures n-gram overlap, not semantic quality
- Higher ROUGE-L doesn't necessarily mean better evaluation results
- ROUGE-L is used only as a training diagnostic

---

## 📄 Citation

If GAD helps your research, please cite the original paper:

```bibtex
@article{ye2025blackboxonpolicydistillationlarge,
  title={Black-Box On-Policy Distillation of Large Language Models},
  author={Tianzhu Ye and Li Dong and Zewen Chi and Xun Wu and Shaohan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2511.10643},
  year={2025},
  url={https://arxiv.org/abs/2511.10643}
}
```

---

## 🙏 Acknowledgments

- Paper [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643)
- [Microsoft LMOps](https://github.com/microsoft/LMOps) official implementation
- [veRL](https://github.com/YTianZHU/verl) framework

---

<div align="center">

Questions or suggestions? Feel free to open an Issue

</div>

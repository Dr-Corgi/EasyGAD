<div align="center">

# EasyGAD

**大语言模型黑盒策略蒸馏实现**

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
  <a href="#-项目简介">简介</a> •
  <a href="#-环境配置">环境配置</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-项目结构">项目结构</a>
</p>

<p align="center">
  简体中文 | <a href="README_EN.md">English</a>
</p>

</div>

---

## 📖 项目简介

本项目实现了论文 [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643) 中提出的 **GAD (Generative Adversarial Distillation)** 算法。

> 💡 如需官方实现，请参考 [microsoft/LMOps/gad](https://github.com/microsoft/LMOps/tree/main/gad) 和 [YTianZHU/verl](https://github.com/YTianZHU/verl)。

GAD 是一种面向大语言模型的知识蒸馏方法。通过对抗训练，学生模型能够在仅访问教师模型黑盒输出的情况下，学习教师模型的输出风格和能力。

### ✨ 为什么选择 EasyGAD？

官方实现需要跨多个仓库和分支才能完成完整实验，本项目提供**一站式解决方案**：

| | 官方实现 | EasyGAD |
|:---:|:---:|:---:|
| 仓库依赖 | 多个外部仓库 | **单一仓库** |
| 分支切换 | 不同阶段切换分支 | **统一分支** |
| 环境配置 | 需分别配置 | **一次配置** |
| 流水线 | 手动串联 | **一键运行** |

### 🎯 核心特性

- 🚀 **开箱即用**：所有依赖内置，克隆即可开始训练
- 🔧 **灵活配置**：支持多阶段独立训练或端到端流水线
- 📦 **完整生态**：集成数据处理、训练、评估全流程
- 🐳 **Docker 支持**：提供预配置镜像，快速上手

### 🎯 训练阶段

| 阶段 | 描述 | Actor 更新 | Critic |
|:---:|:---:|:---:|:---:|
| `seqkd` | SeqKD 基线 | 在教师数据上 SFT | 不使用 |
| `warmup` | GAD 预热 | 在教师数据上 SFT | 判别器 |
| `gad` | GAD 主训练 | PPO + 优势估计 | 判别器 |
| `eval` | 仅评估 | 无更新 | 不使用 |

---

## 🔧 环境配置

### Docker 环境（推荐）

推荐使用 `czwin32768/verl2:v0.2.0-vllm085` 镜像，已预装 `python==3.10.12`, `pytorch==2.6.0`, `vllm==0.8.5`。

```bash
# 拉取并运行容器
docker pull czwin32768/verl2:v0.2.0-vllm085
docker run -it --gpus all czwin32768/verl2:v0.2.0-vllm085

# 进入容器后
cd /tmp
# 将本仓库复制或挂载到 /tmp/gad
cd /tmp/gad
```

### 手动安装

```bash
pip install -e . --no-deps
pip install torchdata rouge-score datasets --upgrade
```

---

## 📦 数据准备

从 HuggingFace 下载教师数据：

```bash
python tools/export_lmsys_parquet.py
```

执行后将生成：
- `/tmp/lmsys_gpt5_chat_filtered_train.parquet` — 训练数据
- `/tmp/lmsys_gpt5_chat_filtered_test.parquet` — 测试数据

---

## 🚀 快速开始

### 一键运行完整流水线

使用 `run_gad_full.sh` 可一键完成 Warmup 和 GAD 两个阶段：

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

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `--model` | 学生模型路径 | 必填 |
| `--reward_model` | 奖励模型路径 | 必填 |
| `--train_files` | 训练数据路径 | `/tmp/lmsys_gpt5_chat_filtered_train.parquet` |
| `--val_files` | 验证数据路径 | `/tmp/lmsys_gpt5_chat_filtered_test.parquet` |
| `--exp_name` | 实验名称 | 必填 |
| `--nnodes` | 节点数量 | 必填 |
| `--warmup_epochs` | Warmup 轮数 | 2 |
| `--gad_epochs` | GAD 训练轮数 | 4 |
| `--resume_step` | 从 Warmup 恢复的检查点步数 | 50 |

### 直接使用预训练模型

如果你已有训练好的 HuggingFace 格式的 Actor 和 Critic 模型，可以直接进行 GAD 训练：

```bash
bash scripts/train/run_gad_direct.sh \
  --actor_path /path/to/pretrained/actor \
  --critic_path /path/to/pretrained/critic \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-direct \
  --nnodes 1
```

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `--actor_path` | 预训练 Actor 模型路径 | 必填 |
| `--critic_path` | 预训练 Critic 模型路径 | 必填 |
| `--train_files` | 训练数据路径 | `/tmp/lmsys_gpt5_chat_filtered_train.parquet` |
| `--val_files` | 验证数据路径 | `/tmp/lmsys_gpt5_chat_filtered_test.parquet` |
| `--exp_name` | 实验名称 | `gad_direct` |
| `--nnodes` | 节点数量 | 1 |

### 分阶段训练

```bash
# 1. Warmup（必需）：初始化判别器并预热学生模型
bash scripts/train/run_warmup.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-warmup-lr1e-6 \
  --nnodes 1

# 2. GAD 训练（必需）：主对抗训练过程
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

# 3. SeqKD（可选）：在教师数据上 SFT，用于对比实验
bash scripts/train/run_seqkd.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --train_files /tmp/lmsys_gpt5_chat_filtered_train.parquet \
  --val_files /tmp/lmsys_gpt5_chat_filtered_test.parquet \
  --exp_name gpt5-chat-filtered-7b-seqkd-lr5e-6 \
  --nnodes 1
```

> 💡 SeqKD 是可选的对比基线，可以直接使用任意预训练模型从 Warmup 开始。

### Python 脚本方式

```bash
# 运行单个阶段
python scripts/run_stage.py --stage seqkd --config configs/seqkd.yaml
python scripts/run_stage.py --stage warmup --config configs/warmup.yaml
python scripts/run_stage.py --stage gad --config configs/gad.yaml

# 运行完整流水线
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

---

## 🧪 模型评估

```bash
# 批量生成
bash scripts/generate/parallel_generate.sh

# 单检查点生成
bash scripts/generate/generate.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --val_data lmsys \
  --ckpt_start 800 --ckpt_end 1200 --ckpt_step 50 \
  --nnodes 1 --ngpus 2
```

---

## 📁 项目结构

```
EasyGAD/
├── verl/                              # 核心 VeRL 库
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py             # 统一训练器
│   │   └── core_algos.py              # PPO、SFT、GRPO 算法
│   └── workers/
│       ├── actor/dp_actor.py          # Actor 实现
│       └── critic/dp_critic.py        # Critic/判别器
├── deepscaler/                        # GAD 工具集
│   ├── globals.py                     # 全局配置
│   ├── system_prompts.py              # 评估系统提示
│   └── rewards/                       # 奖励函数
├── tools/
│   ├── export_lmsys_parquet.py        # 数据准备
│   └── merge_model2hf.py              # 模型转换
├── configs/                           # 配置文件
│   ├── seqkd.yaml / warmup.yaml / gad.yaml / eval.yaml
│   └── pipeline.yaml
└── scripts/                           # 训练脚本
    ├── train/ (run_seqkd.sh, run_warmup.sh, run_gad.sh, run_gad_direct.sh, run_gad_full.sh)
    ├── generate/ (generate.sh, parallel_generate.sh)
    ├── run_stage.py
    └── run_pipeline.py
```

---

## 📝 注意事项

- 训练过程中会记录 ROUGE-L 分数。GAD 的 ROUGE-L 分数可能低于 SeqKD，因为 ROUGE-L 主要衡量 n-gram 重叠，而非深层语义质量
- 更高的 ROUGE-L 分数不一定对应更好的评估结果
- ROUGE-L 仅作为训练诊断指标

---

## 📄 引用

如果 GAD 方法对你的研究有帮助，请引用原论文：

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

## 🙏 致谢

- 论文 [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643)
- [Microsoft LMOps](https://github.com/microsoft/LMOps) 官方实现
- [veRL](https://github.com/YTianZHU/verl) 框架

---

<div align="center">

有问题或建议？欢迎提交 Issue

</div>

# 大语言模型黑盒策略蒸馏

## 项目简介

本项目实现了论文 **Black-Box On-Policy Distillation of Large Language Models** 中提出的 GAD (Generative Adversarial Distillation) 算法。

如需官方实现，请参考 [microsoft/LMOps/gad](https://github.com/microsoft/LMOps/tree/main/gad) 和 [YTianZHU/verl](https://github.com/YTianZHU/verl)。

这是一个**自包含**的仓库，集成了所有依赖项，无需克隆外部仓库。

## 训练阶段概览

本仓库提供**统一流水线**，无需切换分支即可运行所有训练阶段：

| 阶段 | 描述 | Actor更新 | Critic |
|------|------|-----------|--------|
| `seqkd` | SeqKD基线 | 在教师数据上SFT | 不使用 |
| `warmup` | GAD预热 | 在教师数据上SFT | Discriminator |
| `gad` | GAD训练 | PPO + 优势估计 | Discriminator |
| `eval` | 仅评估 | 无更新 | 不使用 |

## 环境配置

### Docker环境（推荐）

推荐使用 `czwin32768/verl2:v0.2.0-vllm085` 镜像，包含 `python==3.10.12, pytorch==2.6.0, vllm==0.8.5`。

```bash
# 拉取并运行Docker容器
docker pull czwin32768/verl2:v0.2.0-vllm085
docker run -it --gpus all czwin32768/verl2:v0.2.0-vllm085

# 在容器内
cd /tmp
# 将本仓库复制或挂载到 /tmp/gad
cd /tmp/gad
```

### 手动安装

```bash
# 安装verl库
pip install -e . --no-deps

# 安装额外依赖
pip install torchdata
pip install rouge-score
pip install datasets --upgrade
```

## 数据准备

从HuggingFace下载教师数据：

```bash
python tools/export_lmsys_parquet.py
```

这将创建：
- `/tmp/lmsys_gpt5_chat_filtered_train.parquet` - 训练数据
- `/tmp/lmsys_gpt5_chat_filtered_test.parquet` - 测试数据

## 快速开始

### 一键训练（推荐）

我们提供统一的脚本，支持一键完成完整的训练流水线。

#### 完整GAD训练（Warmup + GAD）

```bash
bash scripts/train/run_full_gad.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-gad \
  --nnodes 1
```

该脚本自动执行：
1. **Warmup阶段**（2个epoch）- 初始化Discriminator
2. 合并checkpoint模型
3. **GAD阶段**（4个epoch）- 对抗训练

#### 完整SeqKD训练

```bash
bash scripts/train/run_full_seqkd.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-seqkd \
  --nnodes 1
```

#### 仅GAD阶段训练

如果已有warmup checkpoint，可以跳过warmup阶段直接运行GAD：

```bash
# 方式1：指定checkpoint路径
bash scripts/train/run_only_gad.sh \
  --checkpoint /tmp/exp_name/global_step_100 \
  --exp_name gpt5-chat-filtered-7b-gad \
  --nnodes 1

# 方式2：指定工作目录（自动查找最新checkpoint）
bash scripts/train/run_only_gad.sh \
  --work_dir /tmp/exp_name \
  --exp_name gpt5-chat-filtered-7b-gad \
  --nnodes 1
```

#### 可选参数

```bash
# GAD参数
--warmup_epochs 2          # Warmup轮数（默认：2）
--gad_epochs 4             # GAD轮数（默认：4）
--lr 1e-6                  # 学习率（默认：GAD为1e-6，SeqKD为5e-6）
--train_batch_size 256     # 批量大小（默认：256）
--save_freq 50             # 保存频率（默认：50）

# SeqKD参数
--total_epochs 6           # 总轮数（默认：SeqKD为6）
--critic_warmup 10         # Critic预热步数（默认：10）
```

### LoRA训练（参数高效微调）

为节省显存，可使用LoRA版本脚本：

#### 完整GAD训练（LoRA）

```bash
bash scripts/train/run_full_gad_lora.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-gad-lora \
  --nnodes 1 \
  --lora_rank 32
```

#### 完整SeqKD训练（LoRA）

```bash
bash scripts/train/run_full_seqkd_lora.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-seqkd-lora \
  --nnodes 1 \
  --lora_rank 32
```

#### LoRA专属参数

```bash
--lora_rank 32             # LoRA秩（默认：32）
--lr 1e-5                  # LoRA学习率（默认：1e-5，高于全量微调）
```

> **注意**：LoRA训练使用较高的学习率（1e-5 vs 1e-6），并启用参数/优化器卸载以节省显存。

## 评估

生成评估输出：

```bash
bash scripts/generate/parallel_generate.sh
```

或针对单个checkpoint：

```bash
bash scripts/generate/generate.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --val_data lmsys \
  --ckpt_start 800 --ckpt_end 1200 --ckpt_step 50 \
  --nnodes 1 --ngpus 2
```

## 项目结构

```
EasyGAD/
├── verl/                           # 核心VeRL库
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py          # 统一训练器，支持多阶段
│   │   └── core_algos.py           # PPO、SFT、GRPO算法
│   ├── workers/
│   │   ├── actor/dp_actor.py       # Actor，支持SFT/PPO
│   │   └── critic/dp_critic.py     # Critic/Discriminator
│   └── utils/
├── deepscaler/                     # GAD工具
│   ├── globals.py                  # 全局配置
│   ├── system_prompts.py           # 评估系统提示
│   ├── utils.py                    # LLM API工具
│   └── rewards/                    # 奖励函数
├── tools/
│   ├── export_lmsys_parquet.py     # 数据准备
│   └── merge_model2hf.py           # 模型checkpoint转换
├── configs/
│   ├── seqkd.yaml                  # SeqKD阶段配置
│   ├── warmup.yaml                 # Warmup阶段配置
│   ├── gad.yaml                    # GAD训练配置
│   ├── eval.yaml                   # 评估配置
│   └── pipeline.yaml               # 完整流水线配置
├── scripts/
│   ├── train/
│   │   ├── run_full_gad.sh         # 完整GAD流水线（warmup + GAD）
│   │   ├── run_full_gad_lora.sh    # 完整GAD流水线（LoRA版）
│   │   ├── run_full_seqkd.sh       # 完整SeqKD训练
│   │   ├── run_full_seqkd_lora.sh  # 完整SeqKD训练（LoRA版）
│   │   └── run_only_gad.sh         # 仅GAD阶段（从checkpoint启动）
│   ├── generate/
│   │   ├── generate.sh             # 生成脚本
│   │   └── parallel_generate.sh    # 并行生成
│   ├── run_stage.py                # 单阶段运行器
│   └── run_pipeline.py             # 流水线运行器
└── README.md
```

## 训练阶段详解

### 阶段1：Warmup（必需）
- **目的**：初始化Discriminator并预热Student模型
- **方法**：Student采样 → Discriminator评分 → SFT
- **输出**：GAD训练所需的checkpoint

### 阶段2：GAD训练（必需）
- **目的**：主要的对抗训练
- **方法**：Student采样 → Discriminator评分 → PPO
- **输出**：最终训练模型

### 阶段3：评估（可选）
- **目的**：生成输出并评估
- **方法**：仅推理，无训练

### SeqKD（可选基线）
- **目的**：在教师响应上进行SFT基线对比
- **方法**：Teacher forcing GRPO（无discriminator）
- **注意**：GAD训练不需要此阶段。用于消融研究或基线对比。

## 注意事项

- 训练过程中会记录ROUGE-L分数。GAD的ROUGE-L分数可能低于SeqKD，因为ROUGE-L主要捕获n-gram重叠，而非更深层的风格或语义质量。
- 更高的ROUGE-L分数不一定对应更好的自动或人工评估结果。
- ROUGE-L仅作为训练诊断指标，用于验证优化是否正常进行。

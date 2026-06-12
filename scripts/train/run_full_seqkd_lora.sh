#!/bin/bash
# Full SeqKD Training Pipeline with LoRA - Knowledge Distillation via SFT on Teacher Responses
# LoRA mode for parameter-efficient fine-tuning
# Usage: bash run_full_seqkd_lora.sh --model <model_path> --exp_name <name> --nnodes <num_nodes>
set -x

export NCCL_TIMEOUT=36000
export TOKENIZERS_PARALLELISM=true
export SWANLAB_PROJECT='YOUR_PROJECT_NAME'
export SWANLAB_API_KEY='YOUR_SWANLAB_API_KEY'
export SWANLAB_MODE='cloud'
export HYDRA_FULL_ERROR=1

# Default values
TOTAL_EPOCHS=4
SAVE_FREQ=50
TEST_FREQ=50
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=600
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1536
LR=5e-6
TEMPERATURE=0.8
N_SAMPLES=8
CRITIC_WARMUP=10
LORA_RANK=32
WORK_BASE_DIR=/tmp

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --total_epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --critic_warmup)
            CRITIC_WARMUP="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --work_dir)
            WORK_BASE_DIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$EXP_NAME" ] || [ -z "$NNODES" ]; then
    echo "Error: --model, --exp_name, and --nnodes are required"
    exit 1
fi

WORK_DIR=${WORK_BASE_DIR}/${EXP_NAME}
echo "=============================================="
echo "Starting Full SeqKD Training Pipeline with LoRA"
echo "Model: $MODEL_PATH"
echo "Experiment Name: $EXP_NAME"
echo "Nodes: $NNODES"
echo "LoRA Rank: $LORA_RANK"
echo "Work Directory: $WORK_DIR"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Critic Warmup Steps: $CRITIC_WARMUP"
echo "=============================================="

# ============================================
# SeqKD Stage - Supervised Fine-tuning on Teacher Responses
# ============================================
# SeqKD uses teacher forcing GRPO to distill knowledge from teacher responses
# - No KL loss needed (use_kl_loss=False, kl_coef=0.0)
# - Uses vanilla SFT loss for policy
# - Critic warmup is performed internally

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.training_stage=seqkd \
    data.prompt_key=content \
    data.train_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.val_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_RANK} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.val_before_train=True \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${WORK_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Full SeqKD Training Pipeline with LoRA Completed!"
    echo "Checkpoints saved to: ${WORK_DIR}"
    echo "=============================================="
else
    echo "Error: SeqKD training failed!"
    exit 1
fi

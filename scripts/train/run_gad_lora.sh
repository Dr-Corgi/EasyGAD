#!/bin/bash
# GAD Training Stage with LoRA - Main Adversarial Training
# LoRA mode for parameter-efficient fine-tuning
# Usage: ./run_gad_lora.sh --model <model_path> --reward_model <reward_model_path> --exp_name <exp_name> --nnodes <nnodes> --resume_step <step> [--lora_rank <rank>] [--train_files <train_file>] [--val_files <val_file>]
set -x

export NCCL_TIMEOUT=36000

# Default values
TRAIN_FILES="/tmp/lmsys_gpt5_chat_filtered_train.parquet"
VAL_FILES="/tmp/lmsys_gpt5_chat_filtered_test.parquet"
LORA_RANK=32
LORA_ALPHA=32
LR=1e-5

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --reward_model)
            REWARD_MODEL_PATH="$2"
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
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        --train_files)
            TRAIN_FILES="$2"
            shift 2
            ;;
        --val_files)
            VAL_FILES="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$EXP_NAME" ] || [ -z "$NNODES" ] || [ -z "$RESUME_STEP" ]; then
    echo "Error: --exp_name, --nnodes, and --resume_step are required"
    exit 1
fi

# Set default work directory
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="/tmp/${EXP_NAME}"
fi

# Find checkpoint path
CHECKPOINT_PATH="${WORK_DIR}/global_step_${RESUME_STEP}"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

export TOKENIZERS_PARALLELISM=true
export SWANLAB_PROJECT='YOUR_PROJECT_NAME'
export SWANLAB_API_KEY='YOUR_SWANLAB_API_KEY'
# Optional: set to 'local' for offline mode, 'cloud' for cloud mode
export SWANLAB_MODE='cloud'

export HYDRA_FULL_ERROR=1

echo "=============================================="
echo "Starting GAD Training with LoRA"
echo "Experiment Name: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Learning Rate: $LR"
echo "=============================================="

# For LoRA GAD training, we resume from the warmup checkpoint
# The base model paths should be provided for reference (used for tokenizer, config, etc.)
# If not provided, they will be inferred from the checkpoint
if [ -z "$MODEL_PATH" ]; then
    # Try to get model path from checkpoint actor config
    if [ -f "${CHECKPOINT_PATH}/actor/config.json" ]; then
        MODEL_PATH="${CHECKPOINT_PATH}/actor"
    else
        echo "Warning: --model not provided, using checkpoint actor path"
        MODEL_PATH="${CHECKPOINT_PATH}/actor"
    fi
fi

if [ -z "$REWARD_MODEL_PATH" ]; then
    # Try to get reward model path from checkpoint critic config
    if [ -f "${CHECKPOINT_PATH}/critic/config.json" ]; then
        REWARD_MODEL_PATH="${CHECKPOINT_PATH}/critic"
    else
        echo "Warning: --reward_model not provided, using checkpoint critic path"
        REWARD_MODEL_PATH="${CHECKPOINT_PATH}/critic"
    fi
fi

echo "Actor Base Model: $MODEL_PATH"
echo "Critic Base Model: $REWARD_MODEL_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.training_stage=gad \
    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
    data.prompt_key=content \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=256 \
    data.val_batch_size=600 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$REWARD_MODEL_PATH \
    critic.model.lora_rank=${LORA_RANK} \
    critic.model.lora_alpha=${LORA_ALPHA} \
    critic.model.target_modules=all-linear \
    critic.optim.lr=${LR} \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${WORK_DIR} \
    trainer.resume_from_path=${CHECKPOINT_PATH} \
    trainer.resume_mode=resume_path

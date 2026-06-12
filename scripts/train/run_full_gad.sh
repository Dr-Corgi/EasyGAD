#!/bin/bash
# Full GAD Training Pipeline - Warmup + GAD stages combined
# This script runs the complete GAD training in one go
# Usage: bash run_full_gad.sh --model <model_path> --reward_model <reward_model_path> --exp_name <name> --nnodes <num_nodes>
set -x

export NCCL_TIMEOUT=36000
export TOKENIZERS_PARALLELISM=true
export SWANLAB_PROJECT='YOUR_PROJECT_NAME'
export SWANLAB_API_KEY='YOUR_SWANLAB_API_KEY'
export SWANLAB_MODE='cloud'
export HYDRA_FULL_ERROR=1

# Default values
WARMUP_EPOCHS=2
GAD_EPOCHS=4
SAVE_FREQ=50
TEST_FREQ=50
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=600
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1536
LR=1e-6
KL_COEF=0.001
TEMPERATURE=0.8
N_SAMPLES=8

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
        --warmup_epochs)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        --gad_epochs)
            GAD_EPOCHS="$2"
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
        *)
            break
            ;;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$REWARD_MODEL_PATH" ] || [ -z "$EXP_NAME" ] || [ -z "$NNODES" ]; then
    echo "Error: --model, --reward_model, --exp_name, and --nnodes are required"
    exit 1
fi

WORK_DIR=/tmp/${EXP_NAME}
echo "=============================================="
echo "Starting Full GAD Training Pipeline"
echo "Model: $MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Experiment Name: $EXP_NAME"
echo "Nodes: $NNODES"
echo "Work Directory: $WORK_DIR"
echo "=============================================="

# ============================================
# Stage 1: Warmup - Initialize Discriminator and Warm up Student
# ============================================
echo ""
echo "=============================================="
echo "Stage 1: Warmup Phase"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.training_stage=warmup \
    data.prompt_key=content \
    data.train_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.val_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=${REWARD_MODEL_PATH} \
    critic.optim.lr=${LR} \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.val_before_train=True \
    trainer.critic_warmup=10 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${EXP_NAME}_warmup \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=${WARMUP_EPOCHS} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${WORK_DIR}

# Check if warmup succeeded
if [ $? -ne 0 ]; then
    echo "Error: Warmup stage failed!"
    exit 1
fi

echo "Warmup stage completed successfully!"

# ============================================
# Prepare models for GAD stage
# ============================================
echo ""
echo "=============================================="
echo "Preparing models for GAD stage..."
echo "=============================================="

# Find the latest checkpoint from warmup
WARMUP_CHECKPOINT=$(ls -d ${WORK_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -z "$WARMUP_CHECKPOINT" ]; then
    echo "Error: No warmup checkpoint found!"
    exit 1
fi
RESUME_STEP=$(basename $WARMUP_CHECKPOINT | sed 's/global_step_//')
echo "Using warmup checkpoint: $WARMUP_CHECKPOINT (step $RESUME_STEP)"

# Merge actor model
ACTOR_MODEL_PATH=${WARMUP_CHECKPOINT}/actor/huggingface
mkdir -p ${ACTOR_MODEL_PATH}
find ${WARMUP_CHECKPOINT}/actor/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${ACTOR_MODEL_PATH}/ \;
python3 tools/merge_model2hf.py --local_dir ${WARMUP_CHECKPOINT}/actor
if [ $? -ne 0 ]; then
    echo "Error: Failed to merge actor model!"
    exit 1
fi
echo "Actor model prepared at: ${ACTOR_MODEL_PATH}"
ls ${ACTOR_MODEL_PATH}

# Merge critic model
CRITIC_MODEL_PATH=${WARMUP_CHECKPOINT}/critic/huggingface
mkdir -p ${CRITIC_MODEL_PATH}
find ${WARMUP_CHECKPOINT}/critic/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${CRITIC_MODEL_PATH}/ \;
python3 tools/merge_model2hf.py --local_dir ${WARMUP_CHECKPOINT}/critic
if [ $? -ne 0 ]; then
    echo "Error: Failed to merge critic model!"
    exit 1
fi
echo "Critic model prepared at: ${CRITIC_MODEL_PATH}"
ls ${CRITIC_MODEL_PATH}

# ============================================
# Stage 2: GAD - Main Adversarial Training
# ============================================
echo ""
echo "=============================================="
echo "Stage 2: GAD Phase"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.training_stage=gad \
    data.prompt_key=content \
    data.train_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.val_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=${CRITIC_MODEL_PATH} \
    critic.optim.lr=${LR} \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${EXP_NAME}_gad \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=${GAD_EPOCHS} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${WORK_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Full GAD Training Pipeline Completed!"
    echo "Checkpoints saved to: ${WORK_DIR}"
    echo "=============================================="
else
    echo "Error: GAD stage failed!"
    exit 1
fi

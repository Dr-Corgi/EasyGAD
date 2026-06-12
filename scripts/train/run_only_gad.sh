#!/bin/bash
# GAD Stage Only - Run GAD training from an existing warmup checkpoint
# This script skips the warmup phase and starts directly from GAD stage
# Usage: bash run_only_gad.sh --checkpoint <warmup_checkpoint_path> --exp_name <name> --nnodes <num_nodes>
#   Or: bash run_only_gad.sh --work_dir <work_directory> --exp_name <name> --nnodes <num_nodes>
set -x

export NCCL_TIMEOUT=36000
export TOKENIZERS_PARALLELISM=true
export SWANLAB_PROJECT='YOUR_PROJECT_NAME'
export SWANLAB_API_KEY='YOUR_SWANLAB_API_KEY'
export SWANLAB_MODE='cloud'
export HYDRA_FULL_ERROR=1

# Default values
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

CHECKPOINT_PATH=""
WORK_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
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

# Validate required arguments
if [ -z "$EXP_NAME" ] || [ -z "$NNODES" ]; then
    echo "Error: --exp_name and --nnodes are required"
    exit 1
fi

# Determine checkpoint path
if [ -n "$CHECKPOINT_PATH" ]; then
    # User provided explicit checkpoint path
    WARMUP_CHECKPOINT=$CHECKPOINT_PATH
elif [ -n "$WORK_DIR" ]; then
    # Find the latest checkpoint from work directory
    WARMUP_CHECKPOINT=$(ls -d ${WORK_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$WARMUP_CHECKPOINT" ]; then
        echo "Error: No checkpoint found in ${WORK_DIR}"
        exit 1
    fi
else
    # Try default work directory
    WORK_DIR=/tmp/${EXP_NAME}
    WARMUP_CHECKPOINT=$(ls -d ${WORK_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$WARMUP_CHECKPOINT" ]; then
        echo "Error: No checkpoint found. Please provide --checkpoint or --work_dir"
        exit 1
    fi
fi

if [ ! -d "$WARMUP_CHECKPOINT" ]; then
    echo "Error: Checkpoint directory does not exist: ${WARMUP_CHECKPOINT}"
    exit 1
fi

RESUME_STEP=$(basename $WARMUP_CHECKPOINT | sed 's/global_step_//')

echo "=============================================="
echo "Starting GAD Training from Warmup Checkpoint"
echo "Checkpoint: $WARMUP_CHECKPOINT (step $RESUME_STEP)"
echo "Experiment Name: $EXP_NAME"
echo "Nodes: $NNODES"
echo "GAD Epochs: $GAD_EPOCHS"
echo "=============================================="

# ============================================
# Prepare models for GAD stage
# ============================================
echo ""
echo "=============================================="
echo "Preparing models for GAD stage..."
echo "=============================================="

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
# GAD Stage - Main Adversarial Training
# ============================================
echo ""
echo "=============================================="
echo "GAD Phase"
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
    echo "GAD Training Completed!"
    echo "Checkpoints saved to: ${WORK_DIR}"
    echo "=============================================="
else
    echo "Error: GAD stage failed!"
    exit 1
fi

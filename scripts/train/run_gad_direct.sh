#!/bin/bash
# GAD Direct Training - Directly use pretrained HuggingFace models (no warmup needed)
# Usage: ./run_gad_direct.sh --actor_path /path/to/actor --critic_path /path/to/critic --exp_name my_exp --nnodes 1
set -x

export NCCL_TIMEOUT=36000

# Default values
ACTOR_PATH=""
CRITIC_PATH=""
EXP_NAME="gad_direct"
NNODES=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --actor_path)
            ACTOR_PATH="$2"
            shift 2
            ;;
        --critic_path)
            CRITIC_PATH="$2"
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
        *)
            break
            ;;
    esac
done

# Validate required arguments
if [ -z "$ACTOR_PATH" ]; then
    echo "Error: --actor_path is required"
    exit 1
fi

if [ -z "$CRITIC_PATH" ]; then
    echo "Error: --critic_path is required"
    exit 1
fi

# Verify model paths exist
if [ ! -d "$ACTOR_PATH" ]; then
    echo "Error: Actor path does not exist: $ACTOR_PATH"
    exit 1
fi

if [ ! -d "$CRITIC_PATH" ]; then
    echo "Error: Critic path does not exist: $CRITIC_PATH"
    exit 1
fi

echo "Using Actor model: $ACTOR_PATH"
echo "Using Critic model: $CRITIC_PATH"
echo "Experiment name: $EXP_NAME"

export TOKENIZERS_PARALLELISM=true
export SWANLAB_PROJECT='YOUR_PROJECT_NAME'
export SWANLAB_API_KEY='YOUR_SWANLAB_API_KEY'
export SWANLAB_MODE='cloud'

export HYDRA_FULL_ERROR=1

# Create output directory
OUTPUT_DIR=/tmp/${EXP_NAME}
mkdir -p $OUTPUT_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=/tmp/lmsys_gpt5_chat_filtered_train.parquet \
    data.val_files=/tmp/lmsys_gpt5_chat_filtered_test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=600 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$ACTOR_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$CRITIC_PATH \
    critic.optim.lr=1e-6 \
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
    trainer.default_local_dir=$OUTPUT_DIR

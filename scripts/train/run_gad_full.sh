#!/bin/bash
# GAD Full Pipeline - Run warmup and GAD stages sequentially
# Usage: bash run_gad_full.sh --model <model_path> --reward_model <reward_model_path> --train_files <train_file> --val_files <val_file> --exp_name <exp_name> --nnodes <nnodes> [--warmup_epochs <epochs>] [--gad_epochs <epochs>]

set -e

# Default values
TRAIN_FILES="/tmp/lmsys_gpt5_chat_filtered_train.parquet"
VAL_FILES="/tmp/lmsys_gpt5_chat_filtered_test.parquet"
WARMUP_EPOCHS=2
GAD_EPOCHS=4
RESUME_STEP=50  # Default checkpoint step to resume from

# Parse arguments
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
        --train_files)
            TRAIN_FILES="$2"
            shift 2
            ;;
        --val_files)
            VAL_FILES="$2"
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
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model is required"
    exit 1
fi
if [[ -z "$REWARD_MODEL_PATH" ]]; then
    echo "Error: --reward_model is required"
    exit 1
fi
if [[ -z "$EXP_NAME" ]]; then
    echo "Error: --exp_name is required"
    exit 1
fi
if [[ -z "$NNODES" ]]; then
    echo "Error: --nnodes is required"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "GAD Full Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Train Files: $TRAIN_FILES"
echo "Val Files: $VAL_FILES"
echo "Experiment Name: $EXP_NAME"
echo "NNodes: $NNODES"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "GAD Epochs: $GAD_EPOCHS"
echo "Resume Step: $RESUME_STEP"
echo "=========================================="

# Stage 1: Warmup
echo ""
echo "[Stage 1/2] Running Warmup..."
echo ""

bash "$SCRIPT_DIR/run_warmup.sh" \
    --model "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL_PATH" \
    --train_files "$TRAIN_FILES" \
    --val_files "$VAL_FILES" \
    --exp_name "$EXP_NAME" \
    --nnodes "$NNODES" \
    trainer.total_epochs=$WARMUP_EPOCHS

echo ""
echo "[Stage 1/2] Warmup completed!"
echo ""

# Stage 2: GAD Training
echo ""
echo "[Stage 2/2] Running GAD Training..."
echo ""

bash "$SCRIPT_DIR/run_gad.sh" \
    --exp_name "$EXP_NAME" \
    --nnodes "$NNODES" \
    --resume_step "$RESUME_STEP" \
    --train_files "$TRAIN_FILES" \
    --val_files "$VAL_FILES" \
    trainer.total_epochs=$GAD_EPOCHS

echo ""
echo "=========================================="
echo "[Done] GAD Full Pipeline completed!"
echo "=========================================="

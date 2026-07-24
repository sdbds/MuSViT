#!/bin/bash
#SBATCH --job-name=musvit-{I}-{ARCHITECTURE}-{MODEL}-{DATASET}     # Job name
#SBATCH --output=slurm_log/%j.out                                  # Stdout file
#SBATCH --error=slurm_log/%j.err                                   # Stderr file
#SBATCH --partition=dgx                                            # Partition (adapt to your cluster)
#SBATCH --cpus-per-task=1                                          # CPUs per task
#SBATCH --mem=16G                                                  # Memory per node
#SBATCH --gres=gpu:1                                               # Number of GPUs
#SBATCH --gres=shard:16                                            # Estimated GPU memory
#SBATCH --time=2-00:00:00                                          # Time limit (D-HH:MM:SS)

echo "Starting job on `hostname` at `date` $SLURM_IDX"
nvidia-smi

echo "Running in directory $(pwd)"

# Adapt this block to your cluster.
echo "Activating Conda..."
source ../miniconda3/bin/activate
conda activate gpu-env

if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing..."
    pip install uv
fi

# MusViT needs no token: the repositories are public, gated only by accepting their conditions.
# Authenticate once with `huggingface-cli login` on the node, or export HF_TOKEN in your shell.
export WANDB_API_KEY=$(grep WANDB_API_KEY ../../.env | cut -d '=' -f2)

EMBEDDINGS_DIR="difficulty_embeddings/{ARCHITECTURE}/{DATASET}/{MODEL}"

if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
    echo "Generating embeddings..."
    uv run musvit difficulty embeddings --model_name {MODEL} --dataset_name {DATASET} --architecture {ARCHITECTURE}
fi

uv run musvit difficulty run --model_name {MODEL} --dataset_name {DATASET} --architecture {ARCHITECTURE} --fold_idx {I} --epochs 100 --timestamp {TIMESTAMP}

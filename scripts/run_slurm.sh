#!/bin/bash
#SBATCH --job-name=byt5-latin
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2              # Request 2 GPUs (A6000 couple)
#SBATCH --time=48:00:00
#SBATCH --partition=gpu           # Adjust partition name if needed (often 'gpu' or 'short')

# Load necessary modules (adjust based on your cluster, e.g., module load cuda/11.8)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
source .venv/bin/activate

# Echo info
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Set distributed environment variables (if needed by torchrun/accelerate manually, but usually handled)
export OMP_NUM_THREADS=4

# Run training
# Using torchrun for multi-GPU on single node
torchrun --nproc_per_node=2 scripts/train_byt5.py \
    --model_name google/byt5-small \
    --data_file data/train.txt \
    --output_dir checkpoints/byt5-latin-v1 \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_steps 5000 \
    --max_seq_length 512

echo "Training finished"

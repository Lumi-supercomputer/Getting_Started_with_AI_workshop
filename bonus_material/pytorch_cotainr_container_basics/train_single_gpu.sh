#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_single_gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=dev-g  # small-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465001063

# Bind mount user filesystems in the container
export SINGULARITY_BINDPATH="/pfs,/scratch,/projappl,/project,/flash,/appl,$SINGULARITY_BINDPATH"  # or module load singularity-userfilesystems

# Run the training script
srun singularity exec minimal_pytorch.sif python3 train_single_gpu.py

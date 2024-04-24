#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_multi_gpu_ddp_env_setup
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465001063

# Set master address and port for PyTorch process group
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Bind mount user filesystems in the container
export SINGULARITY_BINDPATH="/pfs,/scratch,/projappl,/project,/flash,/appl,$SINGULARITY_BINDPATH"  # or module load singularity-userfilesystems

# Set the mask for "closest" CPU-GPU bindings
c=fe
bind_mask="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Run the training script
srun --cpu-bind=mask_cpu:$bind_mask singularity exec minimal_pytorch.sif python3 train_multi_gpu_ddp_env_setup.py

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
#SBATCH --account=project_465001363

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems)
module use /appl/local/training/modules/AI-20240529

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the containr

# Set master address and port for PyTorch process group
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Workaround MIOpen DB issue when using multiple processes
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Set the mask for "closest" CPU-GPU bindings
c=fe
bind_mask="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Run the training script
srun --cpu-bind=mask_cpu:$bind_mask \
    singularity exec minimal_pytorch.sif \
    bash -c "ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID python3 train_multi_gpu_ddp_env_setup.py"

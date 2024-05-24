#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_multi_gpu_ddp_torchrun
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465001063

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems)
module use /appl/local/training/modules/AI-20240529

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the containr

# Workaround MIOpen DB issue when using multiple processes
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Run the training script
srun singularity exec minimal_pytorch.sif torchrun \
    --standalone \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8  \
    train_multi_gpu_ddp_torchrun.py

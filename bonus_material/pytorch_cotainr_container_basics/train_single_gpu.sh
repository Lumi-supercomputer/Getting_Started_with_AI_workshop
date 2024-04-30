#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_single_gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465001063

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems)
module use /project/project_465001063/modules

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the containr

# Run the training script
srun singularity exec minimal_pytorch.sif python3 train_single_gpu.py

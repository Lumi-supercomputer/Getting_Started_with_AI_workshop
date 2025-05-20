#!/bin/bash
#SBATCH --account=project_465001958
#SBATCH --reservation=AI_workshop_1   # comment this out if the reservation is no longer available
#SBATCH --partition=...
## <!!! ACTION REQUIRED: SPECIFY ADDITIONAL SLURM PARAMETERS HERE!!!>

# Set up the software environment
# NOTE: the loaded module makes relevant filesystem locations available inside the singularity container
#   (/scratch, /project, etc) as well as mounts some important system libraries that are optimized for LUMI
# If you are interested, you can check the exact paths being mounted from
#   /appl/local/containers/ai-modules/singularity-AI-bindings/24.03.lua
module purge
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

CONTAINER=/project/project_465001958/containers/pytorch_transformers.sif

# Some environment variables to set up cache directories
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
FLASH="/flash/${SLURM_JOB_ACCOUNT}"
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$FLASH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

# Path to where the trained model and logging data will go
export OUTPUT_DIR=$SCRATCH/$USER/data/
export LOGGING_DIR=$SCRATCH/$USER/runs/

## <!!! ACTION REQUIRED: RUN THE TRAINING SCRIPT HERE !!!>

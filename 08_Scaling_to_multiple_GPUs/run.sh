#!/bin/bash
#SBATCH --account=project_465002757
#SBATCH --reservation=LUST_AI_Day2   # comment this out if the reservation is no longer available
#SBATCH --partition=...
## <!!! ACTION REQUIRED: SPECIFY ADDITIONAL SLURM PARAMETERS HERE!!!>

# Set up the software environment
# NOTE: the loaded module makes relevant filesystem locations available inside the singularity container
#   (/scratch, /project, etc)
# If you are interested, you can check the exact paths being mounted from
#   /appl/local/laifs/modules/lumi-aif-singularity-bindings/1.0.0.lua
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

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
export MODEL_NAME=gpt-imdb-model-multigpu

## <!!! ACTION REQUIRED: RUN THE TRAINING SCRIPT HERE !!!>

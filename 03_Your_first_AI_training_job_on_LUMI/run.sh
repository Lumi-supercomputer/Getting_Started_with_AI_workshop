#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=...
## <!!! ACTION REQUIRED: SPECIFY ADDITIONAL SLURM PARAMETERS HERE!!!>

# TODO: loading the environment

# Some environment variables to set up cache directories
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$SCRATCH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

# Path to where the trained model and logging data will go
export OUTPUT_DIR=$SCRATCH/$USER/data/
export LOGGING_DIR=$SCRATCH/$USER/runs/
export MODEL_NAME=gpt-imdb-model-${SLURM_JOBID}

## <!!! ACTION REQUIRED: RUN THE TRAINING SCRIPT HERE !!!>


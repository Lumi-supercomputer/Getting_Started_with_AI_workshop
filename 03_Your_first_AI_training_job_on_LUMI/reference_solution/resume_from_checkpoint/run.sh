#!/bin/bash
#SBATCH --account=project_465002757
#SBATCH --reservation=AI_workshop_Day1   # comment this out if the reservation is no longer available
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

# Set up the software environment
# NOTE: the loaded module makes relevant filesystem locations available inside the singularity container
#   (/scratch, /project, etc)
# If you are interested, you can check the exact paths being mounted from
#   /appl/local/laifs/modules/lumi-aif-singularity-bindings/1.0.1.lua
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

set -xv # print the command so that we can verify setting arguments correctly from the logs
srun singularity run $CONTAINER \
    python GPT-neo-IMDB-finetuning.py \
        --model-name gpt-imdb-model \
        --output-path $OUTPUT_DIR \
        --logging-path $LOGGING_DIR \
        --num-workers ${SLURM_CPUS_PER_TASK} \
        --resume  # Comment this for the first run, uncomment to resume in later runs

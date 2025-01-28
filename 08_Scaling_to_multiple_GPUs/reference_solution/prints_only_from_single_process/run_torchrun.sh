#!/bin/bash
#SBATCH --account=project_465001707
#SBATCH --reservation=AI_workshop_2   # comment this out if the reservation is no longer available
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1    # we start a single torchrun process, which will take care of spawning more
#SBATCH --cpus-per-task=56     # 7 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:20:00

# Set up the software environment
# NOTE: the loaded module makes relevant filesystem locations available inside the singularity container
#   (/scratch, /project, etc) as well as mounts some important system libraries that are optimized for LUMI
# If you are interested, you can check the exact paths being mounted from
#   /appl/local/containers/ai-modules/singularity-AI-bindings/24.03.lua
module purge
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

CONTAINER=/project/project_465001707/containers/pytorch_transformers.sif

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

set -xv # print the command so that we can verify setting arguments correctly from the logs

# Since we start only one task with slurm which then starts subprocesses, we cannot use slurm to configure CPU binds.
# Therefore we need to set them up in the Python code itself.

srun singularity exec $CONTAINER \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node=${SLURM_GPUS_PER_NODE} \
             GPT-neo-IMDB-finetuning.py \
             --model-name $MODEL_NAME \
             --output-path $OUTPUT_DIR \
             --logging-path $LOGGING_DIR \
             --num-workers $(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))

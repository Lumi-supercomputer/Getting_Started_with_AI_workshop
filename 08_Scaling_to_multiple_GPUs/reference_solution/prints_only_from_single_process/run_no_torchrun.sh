#!/bin/bash
#SBATCH --account=project_465001958
#SBATCH --reservation=AI_workshop_2   # comment this out if the reservation is no longer available
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8  # we want one process per GPU
#SBATCH --cpus-per-task=7
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
export MODEL_NAME=gpt-imdb-model-multigpu-no-torchrun

set -xv # print the command so that we can verify setting arguments correctly from the logs

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# As opposed to the example in `run_torchrun.sh`, we can set the CPU binds directly via the slurm command, since we have
#  one task per GPU. In this case we do NOT need to set them from within the Python code itself.
srun singularity exec $CONTAINER \
    bash -c "RANK=\$SLURM_PROCID \
             LOCAL_RANK=\$SLURM_LOCALID \
             python GPT-neo-IMDB-finetuning.py \
                --model-name $MODEL_NAME \
                --output-path $OUTPUT_DIR \
                --logging-path $LOGGING_DIR \
                --num-workers ${SLURM_CPUS_PER_TASK}"

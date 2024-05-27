#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=0:15:00
#SBATCH --output=output_ray.txt

# Set up the software environment
module purge
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465001063/containers/pytorch_transformers.sif

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
export MODEL_NAME=gpt-imdb-model

set -xv # print the command so that we can verify setting arguments correctly from the logs

srun singularity exec $CONTAINER python GPT-neo-ray-tune.py \
        --model-name $MODEL_NAME \
        --output-path $OUTPUT_DIR \
        --logging-path $LOGGING_DIR \
        --num-workers ${SLURM_CPUS_PER_TASK}

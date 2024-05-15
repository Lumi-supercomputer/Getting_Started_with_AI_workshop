#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1    # we start a single torchrun process, which will take care of spawning more
#SBATCH --cpus-per-task=56     # 7 cores per GPU
#SBATCH --mem=0
#SBATCH --time=0:15:00


# Set up the software environment
module purge
module use /project/project_465001063/modules
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465001063/containers/pytorch_transformers.sif

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000 0xfe00000000000000 0x0000000000fe0000 0x00000000fe000000 0x00000000000000fe 0x000000000000fe00 0x000000fe00000000 0x0000fe0000000000"

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
export MODEL_NAME=gpt-imdb-model-${SLURM_JOBID}

set -xv # print the command so that we can verify setting arguments correctly from the logs

srun singularity exec $CONTAINER \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node=${SLURM_GPUS_PER_NODE} \
             GPT-neo-IMDB-finetuning.py \
             --model-name $MODEL_NAME \
             --output-path $OUTPUT_DIR \
             --logging-path $LOGGING_DIR \
	     --num-workers $(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE )) \
             --cpu-bind-masks $CPU_BIND_MASKS

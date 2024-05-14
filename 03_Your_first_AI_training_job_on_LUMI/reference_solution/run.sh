#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:10:00

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

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

set -xv # print the python command so that we can verify setting arguments correctly from the logs
srun python pytorch_imdb_gpt.py --datadir $DATADIR --model-name gpt-imdb-model-${SLURM_JOBID} --num_workers ${SLURM_CPUS_PER_TASK}

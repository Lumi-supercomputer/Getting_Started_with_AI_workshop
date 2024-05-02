#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=1:00:00

export EBU_USER_PREFIX="/project/${SLURM_JOB_ACCOUNT}/lukaspre/EasyBuild"
module purge
module load LUMI
module load PyTorch

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$SCRATCH/data/users/$USER
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$SCRATCH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

set -xv
srun singularity exec $SIF python pytorch_imdb_gpt.py --datadir $DATADIR --model-name gpt-imdb-model-${SLURM_JOBID} --num_workers ${SLURM_CPUS_PER_TASK}

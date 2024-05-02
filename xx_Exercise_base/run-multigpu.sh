#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
##SBATCH --mem-per-gpu=60G
#SBATCH --mem=0
#SBATCH --time=1:00:00

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$SCRATCH/data/users/$USER
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$SCRATCH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

set -xv
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS

srun bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python pytorch_imdb_gpt_multigpu.py --datadir $DATADIR --model-name gpt-imdb-model-${SLURM_JOBID} --num_workers ${SLURM_CPUS_PER_TASK}"


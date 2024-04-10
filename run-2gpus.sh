#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=0:20:00

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache
#export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns

mkdir -p $TORCH_HOME $HF_HOME $MLFLOW_TRACKING_URI

#export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

set -xv
#srun python3 $* --datadir $DATADIR --num_workers ${SLURM_CPUS_PER_TASK} --rank ${SLURM_PROCID} --num_processes ${SLURM_NTASKS}
srun torchrun --standalone --nnodes=1 --nproc_per_node=2 $* --datadir $DATADIR --num_workers 7


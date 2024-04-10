#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=0:20:00

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache
#export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns  # this causes an MLFlow exception during imdb training

mkdir -p $TORCH_HOME $HF_HOME $MLFLOW_TRACKING_URI

set -xv
python3 $* --datadir $DATADIR --num_workers ${SLURM_CPUS_PER_TASK}

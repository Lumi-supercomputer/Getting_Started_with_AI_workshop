#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_multi_gpu_hf_trainer
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --account=project_465001063

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems)
module use /appl/local/training/modules/AI-20240529

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the containr

# Workaround MIOpen DB issue when using multiple processes
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Setup Hugging Face cache and data directories
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
export DATADIR=$SCRATCH/data/users/$USER
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$SCRATCH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

srun singularity exec hf_env_container.sif \
    accelerate launch \
        --config_file=accelerate_hf_trainer_config.yaml \
        --num_machines=1 \
        --num_processes=${SLURM_GPUS_PER_NODE} \
        --machine_rank=0 \
        train_hf_imdb_gpt.py \
            --datadir ${DATADIR} \
            --model-name gpt-imdb-model-${SLURM_JOBID} \
            --num_workers $(expr ${SLURM_CPUS_PER_TASK} / ${SLURM_GPUS_PER_NODE}) \

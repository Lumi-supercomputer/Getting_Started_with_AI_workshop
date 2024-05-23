#!/bin/bash
#SBATCH --account=project_465001063
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8  # we want one process per GPU
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00


# Set up the software environment
# NOTE: these modules will be available from the LUMI system stack after July 2024 and the "module use" line will no longer work
module purge
module use /project/project_465001063/modules
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
export MODEL_NAME=gpt-imdb-model-${SLURM_JOBID}

set -xv # print the command so that we can verify setting arguments correctly from the logs

# Ensure that RCCL uses the high-speed interconnect instead of something slow like TCP
export NCCL_SOCKET_IFNAME=hsn

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$WORLD_SIZE

# As opposed to the example in `run_torchrun.sh`, we can set the CPU binds directly via the slurm command, since we have
#  one task per GPU. In this case we do NOT need to set them from within the Python code itself.

# Set up the CPU bind masks (can only be used with full node runs (standard-g or small-g with slurm argument `--exclusive`))
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# tell slurm to configure the cpu binds specified by the mask, additional option v prints to configuration to the logs
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS \
    singularity exec $CONTAINER \
    bash -c "RANK=\$SLURM_PROCID \
             LOCAL_RANK=\$SLURM_LOCALID \
             python GPT-neo-IMDB-finetuning.py \
                --model-name $MODEL_NAME \
                --output-path $OUTPUT_DIR \
                --logging-path $LOGGING_DIR \
                --num-workers ${SLURM_CPUS_PER_TASK}"

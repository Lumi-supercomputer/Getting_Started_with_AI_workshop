#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_multi_node_hf_trainer
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=00:45:00
#SBATCH --account=project_465001063

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems)
module use /appl/local/training/modules/AI-20240529

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the containr

# Bind mount CPE bits to enable RCCL communication using the Slingshot interconnect via the aws-ofi-rccl plugin
module load singularity-CPEbits  # corresponds to specifying --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 when running the container

# Set network interfaces to be used by RCCL to workaround RCCL failing to auto-detect the correct interface
export NCCL_SOCKET_IFNAME=hsn

# Tune when GPU Direct RDMA between NIC and GPU is used
export NCCL_NET_GDR_LEVEL=PHB

# Setup Hugging Face cache and data directories
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
export DATADIR=$SCRATCH/data/users/$USER
export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=$SCRATCH/hf-cache
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

# Launch the training using accelerate
# Note: --machine_rank must be evaluated on each node, hence the LAUNCH_CMD setup
export LAUNCH_CMD="
    accelerate launch \
        --config_file=accelerate_hf_trainer_config.yaml \
        --num_machines=${SLURM_NNODES} \
        --num_processes=$(expr ${SLURM_NNODES} \* ${SLURM_GPUS_PER_NODE}) \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=$(hostname -i) \
        train_hf_imdb_gpt.py \
            --datadir ${DATADIR} \
            --model-name gpt-imdb-model-${SLURM_JOBID} \
            --num_workers $(expr ${SLURM_CPUS_PER_TASK} / ${SLURM_GPUS_PER_NODE})\
    "
srun singularity exec pytorch_transformers.sif bash -c "${LAUNCH_CMD}"

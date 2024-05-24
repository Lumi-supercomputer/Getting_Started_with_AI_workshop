#!/usr/bin/env -S bash -e
#SBATCH --job-name=train_multi_node_ddp_torchrun
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --account=project_465001063

# Load custom modules that are not (yet) part of the central LUMI stack (singularity-userfilesystems, singularity-CPEbits)
module use /appl/local/training/modules/AI-20240529

# Bind mount user filesystems in the container
module load singularity-userfilesystems  # corresponds to specifying --bind /pfs,/scratch,/projappl,/project,/flash,/appl when running the container

# Bind mount CPE bits to enable RCCL communication using the Slingshot interconnect via the aws-ofi-rccl plugin
module load singularity-CPEbits  # corresponds to specifying --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 when running the container

# Set network interfaces to be used by RCCL to workaround RCCL failing to auto-detect the correct interface
export NCCL_SOCKET_IFNAME=hsn

# Tune when GPU Direct RDMA between NIC and GPU is used
export NCCL_NET_GDR_LEVEL=PHB

# Workaround MIOpen DB issue when using multiple processes
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Uncomment to set NCCL debug output to check correct use of aws-ofi-rccl (these are VERY verbose)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,COLL

# Run the training script
srun singularity exec minimal_pytorch.sif torchrun \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$(hostname):29500" \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8  \
    train_multi_gpu_ddp_torchrun.py

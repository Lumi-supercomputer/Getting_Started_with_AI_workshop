# 10 Extreme scale AI
## LLM hands-on exercises
### 1. Setting some run scripts is a great idea!

Here's how to setup a wrapper script:

```
cat > run.sh << EOF
#!/bin/bash -e

# Report affinity
echo "Rank \$SLURM_PROCID --> \$(taskset -p \$\$)"

# Report GPUs
if [ \$SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
else
  sleep 2
fi

# Start conda environment inside the container
\$WITH_CONDA

# Setting the caches relevant to our application.
export TORCH_HOME=/workdir/torch-cache
export HF_HOME=/workdir/hf-cache
export TOKENIZERS_PARALLELISM=false

# Tell RCCL to use only Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# Tell MIOpen where to store its cache
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH

if [ \$SLURM_LOCALID -eq 0 ] ; then
  rm -rf \$MIOPEN_USER_DB_PATH
  mkdir -p \$MIOPEN_USER_DB_PATH    
else
  sleep 2
fi

# export NCCL_DEBUG=INFO 
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export NCCL_DEBUG_FILE=/tmp/$(whoami)-rccl-rank\$SLURM_PROCID.txt

# Translate SLURM environment 

export MASTER_PORT=25900
export WORLD_SIZE=\$SLURM_NPROCS
export LOCAL_WORLD_SIZE=8
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

set -x

# Run application
eval "\$@"

EOF
chmod +x run.sh
```

Let's take a look on what is going on here from top to bottom:
* We leverage the `taskset` tool to report the affinity of the current process. This allows us to verify we are getting the affinity we expect.
* Then, we report the GPUs available using rocm-smi. This is a smoke test that the GPUs are up and running. We do this only for the first rank in a node - that rank will have `SLURM_LOCALID` set to `0`.
* Then, we setup our conda environment as well as a few other environment variables to control the Pytorch and HuggingFace caches for our application.
* Then we configure RCCL to use the high-speed interfaces as well as GPU RDMA.
* Next step is the MIOpen cache. We also have the first rank in each node creating the cache folder. Note that, this is not used by our LLM application as it doesn't use MIOpen kernels. However, it doesn't do any harm and we'll keep you covered for other models you might want to train.
* Then, there are a few RCCL environment variables that you may chose to uncomment so as to get logging of the RCCL activity.
* Next, we translate the SLURM environment to something that Pytorch distributed module understands.
* Finally, the arguments of the run scrips are expanded and executed.
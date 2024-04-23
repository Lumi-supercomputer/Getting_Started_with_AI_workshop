# PyTorch cotainr container basics on LUMI

This is a short introduction to building a PyTorch container with cotainr and running it on LUMI as a container with a minimal interface to the LUMI host software stacks.

NOTE: This assumes `module use /project/project_465001063/...` with installed modules:

- An updated installation of the `cotainr` module that sets `--system=lumi-g` to use the LUMI ROCm base image (/appl/local/`containers/sif-images/lumi-rocm-rocm-5.6.1.sif`)
- The new `singularity-userfilesystems` module that bind mounts user file system paths, i.e. `/project`, `/scratch`, and `/flash`.
- The new `singularity-CPEbits` module that bind mounts the parts of the Cray Programming Environment from the host that are missing in the official LUMI container images due to license restrictions imposed by HPE.

## 1. Specify your conda/pip environment

To get started, specify all needed Python packages + dependencies in a [conda environment yaml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Preferably, specify as many packages as possible as conda packages instead of pip packages as [recommended by the conda developers](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment). Whenever AMD GPU acceleration is needed for a given Python package, it must be installed in a version that is compiled against ROCm. A minimal PyTorch conda environment file may look like `minimal_pytorch.yml`. As a minimum, pin the versions of the packages (as done in `minimal_pytorch.yml`), or even better, also pin the build numbers to allow for maximal reproducibility.

### NOTE: Using the LUMI PyTorch container instead

If you only need a basic PyTorch setup, and don't need to install any extra Python packages yourself, you may use the LUMI PyTorch container and skip building a custom container. If you use the LUMI PyTorch container, you have to be aware that it the conda environment in that container is not automatically activated when running the container as is the case with containers build using cotainr. Instead you have to manually activate it by running `$WITH_CONDA` as the first thing in the container.

## 2. Build a container using cotainr

Next, build a container that includes the conda environment based on the official LUMI ROCm base image using cotainr.

```bash
#module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml  # or use --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif instead of --system=lumi-g
```

If you need to change something in the conda environment, update the `minimal_pytorch.yml` file and rebuild the container. To avoid putting stress on the login-nodes, you may want to consider running cotainr non-interactively on a compute node instead of the login nodes, e.g.

```bash
srun --output=cotainr.out --error=cotainr.err --account=project_465001063 --time=00:30:00 --mem=64G --cpus-per-task=32 --partition=debug cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml --accept-licenses
```

More details about building conda/pip environment containers using cotainr in the [cotainr documentation](https://cotainr.readthedocs.io/en/latest/user_guide/conda_env.html).

### NOTE: Installing software not managed by conda/pip
proot

## 3. Run the MNIST training example on a single GPU (GCD)

Consider the MNIST model and training setup defined `mnist_model.py`. In order to run this training on a single CGD on LUMI, you just need to run the `train_single_gpu.py` script using the `minimal_pytorch.sif` container on a LUMI-G node. The `train_single_gpu.sh` SLURM batch script provides an example of how to launch this training using a single node in the `small-g` partition.

### NOTE: Getting access to the user file systems in the container
module load singularity-userfilesystems

## 4. Scale to multiple GPUs (GCDs) within a single node using DDP

In order to scale the training to multiple GCDs on LUMI, the Python training code must be adapted to use some distributed training strategy, e.g. Distributed Data Parallel (DDP). When launching such a distributed training, the distributed environment, i.e. the processes handling the training and their communication, must be setup correctly. We recommend to launch one process per GCD. Since there are 8 GCDs per LUMI-G node, this results in 8 processes that need to communicate. To get the shortest communication path and thus, optimal performance, these processes must be bound to the CPU cores that are closest (in terms for communication latency) to their corresponding GCDs. 

The `train_multi_gpu_ddp_env_setup.py` Python script together with the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script provides an example of how to launch this training using a single node in the `standard-g` partition, setting up the distributed environment manually based on SLURM environment variables.

## 5. Scale to multiple nodes
To scale to multiple nodes, you may simple request more nodes in the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script and the PyTorch DDP training should automatically scale to all available nodes. However, in practice a few other things are needed to get optimal performance and workaround some know issues with running such training on multiple nodes on LUMI:

* To have hardware acceleration for inter node communication via RCCL (`backend="nccl"` in PyTorch's `init_process_group`) over the Slingshot 11 interconnect, the `aws-ofi-rccl` plugin that bridges RCCL with libfabric (which is the transport layer used with Slingshot 11) must be available. The `asw-ofi-rccl` plugin is installed in the LUMI ROCm base image. Additionally some of the HPE Slingshot libraries must also be available for this to work. Due to licenses restrictions imposed by HPE, we cannot include these libraries in the container. Instead they must be bind mounted from LUMI at runtime. A shortcut to obtaining these bind mounts are `module load singularity-CPEbits`. In addition to setting these bind mounts, one must also explicitly set the environment variable `NCCL_SOCKET_IFNAME=hsn` to workaround RCCL not being able to automatically identify the correct network interfaces to use.
* To workaround a problem with the accessing a single MIOpen SQLite database on the lustre filesystems from multiple nodes, one must set the environment variables `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to a node local location, e.g. a folder in /tmp.

This entire setup is shown in `train_multi_node_ddp_env_setup.sh`.

### NOTE: Checking correct load of aws-ofi-rccl plugin
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

## Workarounds that may be needed
- Warm starting GPU using rocm-smi
- Multiple dataloader workers - no fork safe cxi

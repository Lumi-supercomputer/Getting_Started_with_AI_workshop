# PyTorch cotainr container basics on LUMI

This is a short introduction to building a PyTorch container with cotainr and running it on LUMI as a container with a minimal interface to the LUMI host software stacks. A general understanding of the architecture of LUMI is assumed.

> [!WARNING]
> To run these examples, it is assumed that you `module use /appl/local/training/modules/AI-20240529` with installed modules:
>
> - An updated installation of the `cotainr` module that sets `--system=lumi-g` to use the LUMI ROCm base image (/appl/local/`containers/sif-images/lumi-rocm-rocm-6.0.3.sif`)
> - The new `singularity-userfilesystems` module that bind mounts user file system paths, i.e. `/project`, `/scratch`, and `/flash`.
> - The new `singularity-CPEbits` module that bind mounts the parts of the Cray Programming Environment from the host that are missing in the official LUMI container images due to license restrictions imposed by HPE.
>
> After the AI workshop and the LUMI maintenance break in August/September, these modules will hopefully be available in the central LUMI stack.

## 1. Specify your conda/pip environment

To get started, specify all needed Python packages + dependencies in a [conda environment yaml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Preferably, specify as many packages as possible as conda packages instead of pip packages as [recommended by the conda developers](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment). Whenever AMD GPU acceleration is needed for a given Python package, it must be installed in a version that is compiled against ROCm. A minimal PyTorch conda environment file may look like `minimal_pytorch.yml`. As a minimum, pin the versions of the packages (as done in `minimal_pytorch.yml`), or even better, also pin the build numbers to allow for maximal reproducibility.

> [!NOTE]
>If you only need a basic PyTorch setup, and don't need to install any extra Python packages yourself, you may use the LUMI PyTorch container (found under /appl/local/containers/sif-images on LUMI) and skip building a custom container. If you use the LUMI PyTorch container, you have to be aware that the conda environment in that container is not automatically activated when running the container, as is the case with containers built using cotainr. Instead you have to manually activate it by running `$WITH_CONDA` as the first thing in the container.

## 2. Build a container using cotainr

Next, build a container that includes the conda environment based on the official LUMI ROCm container base image using cotainr.

```bash
module load CrayEnv cotainr
cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml  # or use --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.0.3.sif instead of --system=lumi-g when using cotainr from the LUMI stack
```

If you need to change something in the conda environment, update the content of `minimal_pytorch.yml` and rebuild the container. To avoid putting stress on the login-nodes, you may want to consider running cotainr non-interactively on a compute node instead of the login nodes, e.g.

```bash
srun --output=cotainr.out --error=cotainr.err --account=project_465001063 --time=00:30:00 --mem=64G --cpus-per-task=32 --partition=debug cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml --accept-licenses
```

More details about building conda/pip environment containers using cotainr may be found in the [cotainr documentation](https://cotainr.readthedocs.io/en/latest/user_guide/conda_env.html) and the [LUMI Docs cotainr documentation](https://docs.lumi-supercomputer.eu/software/containers/singularity/#building-containers-using-the-cotainr-tool).

> [!NOTE]
> Currently, cotainr only supports conda environments as a way to specify software to install when building a container using cotainr. While most AI related software can be installed via conda/pip there may be some packages that cannot. To install such packages in the container, the container must be built using other tools than cotainr. On LUMI, SingularityCE + proot may be used to build a container from a Singularity definition file. An example of such a SingularityCE + proot build may be found in [the LUMI-training-materials archive](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20240502/09_Containers/#extending-the-container-with-the-singularity-unprivileged-proot-build). More details about building containers in general for LUMI may be found in the [LUMI Docs containers section](https://docs.lumi-supercomputer.eu/software/containers/singularity/).

## 3. Run a PyTorch training on a single GPU (GCD)

Consider the MNIST model and training setup defined `mnist_model.py`. In order to run this training on a single CGD on LUMI, you just need to run the `train_single_gpu.py` script using the `minimal_pytorch.sif` container on a LUMI-G node. The `train_single_gpu.sh` SLURM batch script provides an example of how to launch this training using a single node in the `small-g` partition.

> [!NOTE]
> In order to get access to user file systems, i.e. project folders under /project, /scratch, and /flash, on LUMI, you need to bind mount these into the container at runtime. Since these paths are set using symlinks on LUMI, you also need to bind mount the "true" file systems since Singularity does not follow symlinks in bind mounted paths. A shortcut to getting these binds right is `module load singularity-userfilesystems`.

## 4. Scale to multiple GPUs (GCDs) within a single node using DDP

In order to scale the training to multiple GCDs in a single node on LUMI, the Python training code must be adapted to use some distributed training strategy, e.g. Distributed Data Parallel (DDP). When launching such a distributed training, the distributed environment, i.e. the handling of processes and their communication, must be setup correctly. We recommend to launch one process per GCD and use RCCL for communication between GPUs/GCDs. Since there are 8 GCDs per LUMI-G node, this results in 8 processes that need to communicate via RCCL. Optionally, to get the shortest communication path, and thus optimal performance, these processes must be bound to the CPU cores that are closest (in terms for communication latency) to their corresponding GCDs. There are multiple ways to setup such a distributed training, e.g.

- **SLURM environment setup**: Let SLURM handle the process management and bindings, and use the relevant SLURM environment variables in your Python script when calling `torch.distributed.init_process_group()`. The `train_multi_gpu_ddp_env_setup.py` Python script together with the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script provides an example of how to launch a training using a single node in the `standard-g` partition based on this approach.
- **torchrun setup**: Have SLURM launch a single process with all CPU cores and GPUs allocated for the torch elastic launcher. Have `torchrun` manage the training processes and manually set the correct bindings from within your Python script based on the environment variables set by `torchrun`. The `train_multi_gpu_ddp_torchrun.py` Python script together with the `train_multi_gpu_ddp_torchrun.sh` SLURM batch script provides an example of how to launch a training using a single node in the `standard-g` partition based on this approach.

> [!NOTE]
> To workaround a problem with the accessing a single MIOpen SQLite database (used by ROCm) on the Lustre file systems from multiple nodes, one must (at least in this example) set the environment variables `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to a node local location, e.g. a folder in /tmp.

> [!NOTE]
> The PyTorch (GPU) device is automatically chosen based on the ROCR_VISIBLE_DEVICES environment variable if not explicitly defined in the Python code, e.g. via `torch.device()`. Ideally, SLURM should be able to correctly set ROCR_VISIBLE_DEVICES for each rank if requesting a single GPU per rank. However, as of 20240524, this is not the case on LUMI since GPUs are constrained using cgroups. See <https://bugs.schedmd.com/show_bug.cgi?id=17875> for more details. Consequently, we need to manually set `ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID` just before running our Python training script when using the **SLURM environment setup** approach.

## 5. Scale to multiple nodes

To scale to multiple nodes, you may simply request more nodes in the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script when using the **SLURM environment setup** approach. When using the **torchrun setup** approach, you also need to setup the torchrun randezvous correctly. The PyTorch DDP training in `train_multi_gpu_ddp_env_setup.py`/`train_multi_gpu_ddp_torchrun.py` should automatically scale to all available nodes. However, in practice, a few other things are needed to get optimal performance when running such a training on multiple nodes on LUMI:

- To have hardware acceleration for inter node communication via RCCL (`backend="nccl"` in PyTorch's `init_process_group`) over the Slingshot 11 interconnect, the `aws-ofi-rccl` plugin that bridges RCCL with libfabric (which is the transport layer used with Slingshot 11) must be available. The `asw-ofi-rccl` plugin is installed in the LUMI ROCm base image. Additionally some of the HPE Slingshot libraries must also be available for this to work. Due to licenses restrictions imposed by HPE, we cannot (yet) include these libraries in the container. Instead they must be bind mounted from LUMI at runtime. A shortcut to obtaining these bind mounts are `module load singularity-CPEbits`.
- When using the `aws-ofi-rccl` plugin, one must also explicitly set the environment variable `NCCL_SOCKET_IFNAME=hsn` to workaround RCCL not being able to automatically identify the correct network interfaces to use.
- For increased performance for large message sizes, it may be necessary to tune when RCCL uses GPU Direct RDMA between a NIC and a GPU by setting the environment variable `NCCL_NET_GDR_LEVEL=PHB`. More details about `NCCL_NET_GDR_LEVEL` in [the Nvidia docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level). Note, though, that this setting is known to cause crashes/hangs when using more than 256 nodes.
- To have PyTorch attempt to automatically handle hangs in RCCL communication, you may set the environment variable `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (the default value set by PyTorch >= 1.12). In case of a hang in RCCL communication, this should bring down the communication and stop the training job, avoiding wasted resources from indefinitely hanging jobs. When used with the **SLURM environment setup**, the training is terminated in case of a hang. When used with the **torchrun setup**, torchrun attempts to restart the training, so remember to use [checkpointing](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) to have it restart from the latest checkpoint (checkpointing is not used in the MNIST toy examples given here...). When setting `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, you may need to tune the `timeout` argument given to `torch.distributed.init_process_group()` as it determines when a "hang" is occurring. More details in [this PyTorch issue](https://github.com/pytorch/pytorch/issues/46874).
- When using multiple PyTorch DataLoader workers, you need to set the Python/PyTorch multiprocessing spawn method to "spawn" instead of "fork", since forking is not safe with the Slingshot 11, and may lead to crashes. Note that the call to `torch.multiprocessing.set_start_method("spawn")` [should be guarded in a `if __name__ == '__main__'` clause](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-start-methods). Alternatively, you must specify `workers=0` for the DataLoader to keep data load in the main process.

All of these are optional. If the `aws-ofi-rccl` plugin is not used, RCCL falls back to communication via TCP/IP sockets, which may significantly degrade performance.

This entire setup is shown in the `train_multi_node_ddp_env_setup.sh`/`train_multi_node_ddp_torchrun.sh` SLURM batch scripts which launch the training on 4 nodes in standard-g for a total of 4 x 8 = 32 GCDs used in the training.

> [!TIP]
> To check if the `aws-ofi-rccl` plugin is being used in the training, you may set the following NCCL/RCCL debug environment variables and inspect the standard output from the training.
>```bash
>export NCCL_DEBUG=INFO
>export NCCL_DEBUG_SUBSYS=INIT,COLL
>```
> The standard output includes lines like `NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0` if the `aws-ofi-rccl` plugin has been loaded correctly.

# PyTorch cotainr container basics on LUMI

This is a short introduction to building a PyTorch container with cotainr and running it on LUMI as a container with a minimal interface to the LUMI host software stacks.

### WARNING: Custom LUMI modules are used in this example

This assumes `module use /project/project_465001063/EB/modules` with installed modules:

- An updated installation of the `cotainr` module that sets `--system=lumi-g` to use the LUMI ROCm base image (/appl/local/`containers/sif-images/lumi-rocm-rocm-5.6.1.sif`)
- The new `singularity-userfilesystems` module that bind mounts user file system paths, i.e. `/project`, `/scratch`, and `/flash`.
- The new `singularity-CPEbits` module that bind mounts the parts of the Cray Programming Environment from the host that are missing in the official LUMI container images due to license restrictions imposed by HPE.

After the AI workshop and the LUMI maintenance break in June, these module will hopefully be available in the central LUMI stack.

## 1. Specify your conda/pip environment

To get started, specify all needed Python packages + dependencies in a [conda environment yaml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Preferably, specify as many packages as possible as conda packages instead of pip packages as [recommended by the conda developers](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment). Whenever AMD GPU acceleration is needed for a given Python package, it must be installed in a version that is compiled against ROCm. A minimal PyTorch conda environment file may look like `minimal_pytorch.yml`. As a minimum, pin the versions of the packages (as done in `minimal_pytorch.yml`), or even better, also pin the build numbers to allow for maximal reproducibility.

### NOTE: Using the LUMI PyTorch container instead

If you only need a basic PyTorch setup, and don't need to install any extra Python packages yourself, you may use the LUMI PyTorch container and skip building a custom container. If you use the LUMI PyTorch container, you have to be aware that the conda environment in that container is not automatically activated when running the container, as is the case with containers built using cotainr. Instead you have to manually activate it by running `$WITH_CONDA` as the first thing in the container.

## 2. Build a container using cotainr

Next, build a container that includes the conda environment based on the official LUMI ROCm container base image using cotainr.

```bash
module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml  # or use --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif instead of --system=lumi-g when using cotainr from LUMI/23.03
```

If you need to change something in the conda environment, update the content of `minimal_pytorch.yml` and rebuild the container. To avoid putting stress on the login-nodes, you may want to consider running cotainr non-interactively on a compute node instead of the login nodes, e.g.

```bash
srun --output=cotainr.out --error=cotainr.err --account=project_465001063 --time=00:30:00 --mem=64G --cpus-per-task=32 --partition=debug cotainr build minimal_pytorch.sif --system=lumi-g --conda-env=minimal_pytorch.yml --accept-licenses
```

More details about building conda/pip environment containers using cotainr may be found in the [cotainr documentation](https://cotainr.readthedocs.io/en/latest/user_guide/conda_env.html) and the [LUMI Docs cotainr documentation](https://docs.lumi-supercomputer.eu/software/containers/singularity/#building-containers-using-the-cotainr-tool).

### NOTE: Installing software not managed by conda/pip

Currently, cotainr only supports conda environments as a way to specify software to install when building a container using cotainr. While most AI related software can be installed via conda/pip there may be some packages that cannot. To install such packages in the container, the container must be built using other tools than cotainr. On LUMI, SingularityCE + proot may be used to build a container from a Singularity definition file. An example of such a SingularityCE + proot build may be found in [this LUMI-training-materials preview](https://klust.github.io/LUMI-training-materials/2day-20240502/09_Containers/#extending-the-container-with-the-singularity-unprivileged-proot-build). More details about building containers in general for LUMI may be found in the [LUMI Docs containers section](https://docs.lumi-supercomputer.eu/software/containers/singularity/).

## 3. Run a PyTorch training on a single GPU (GCD)

Consider the MNIST model and training setup defined `mnist_model.py`. In order to run this training on a single CGD on LUMI, you just need to run the `train_single_gpu.py` script using the `minimal_pytorch.sif` container on a LUMI-G node. The `train_single_gpu.sh` SLURM batch script provides an example of how to launch this training using a single node in the `small-g` partition.

### NOTE: Getting access to the user file systems in the container

In order to get access to user file systems, i.e. project folders under /project, /scratch, and /flash, on LUMI, you need to bind mount these into the container at runtime. Since these paths are set using symlinks on LUMI, you also need to bind mount the "true" file systems since Singularity does not follow symlinks in bind mounted paths. A (soon to be) shortcut to getting this binds right is `module load singularity-userfilesystems`.

## 4. Scale to multiple GPUs (GCDs) within a single node using DDP

In order to scale the training to multiple GCDs in a single node on LUMI, the Python training code must be adapted to use some distributed training strategy, e.g. Distributed Data Parallel (DDP). When launching such a distributed training, the distributed environment, i.e. the handling of processes and their communication, must be setup correctly. We recommend to launch one process per GCD and use RCCL for communication between GPUs/GCDs. Since there are 8 GCDs per LUMI-G node, this results in 8 processes that need to communicate via RCCL. To get the shortest communication path, and thus optimal performance, these processes must be bound to the CPU cores that are closest (in terms for communication latency) to their corresponding GCDs.

The `train_multi_gpu_ddp_env_setup.py` Python script together with the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script provides an example of how to launch this training using a single node in the `standard-g` partition, setting up the distributed environment manually based on SLURM environment variables.

### NOTE: GPU visibility on LUMI and choosing the right PyTorch device

The PyTorch (GPU) device is automatically chosen based on the ROCR_VISIBLE_DEVICES environment variable if not explicitly defined in the Python code, e.g. via `torch.device()`. Ideally, SLURM should be able to correctly set ROCR_VISIBLE_DEVICES for each rank if requesting a single GPU per rank. However, as of 20240424, this is not the case on LUMI since GPUs are constrained using cgroups. See <https://bugs.schedmd.com/show_bug.cgi?id=17875> for more details. Consequently, we need to manually set `ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID` just before running our Python training script.

## 5. Scale to multiple nodes

To scale to multiple nodes, you may simple request more nodes in the `train_multi_gpu_ddp_env_setup.sh` SLURM batch script. The PyTorch DDP training in `train_multi_gpu_ddp_env_setup.py` should automatically scale to all available nodes. However, in practice, a few other things are needed to get optimal performance and workaround some know issues with running such a training on multiple nodes on LUMI:

- To have hardware acceleration for inter node communication via RCCL (`backend="nccl"` in PyTorch's `init_process_group`) over the Slingshot 11 interconnect, the `aws-ofi-rccl` plugin that bridges RCCL with libfabric (which is the transport layer used with Slingshot 11) must be available. The `asw-ofi-rccl` plugin is installed in the LUMI ROCm base image. Additionally some of the HPE Slingshot libraries must also be available for this to work. Due to licenses restrictions imposed by HPE, we cannot include these libraries in the container. Instead they must be bind mounted from LUMI at runtime. A (soon to be) shortcut to obtaining these bind mounts are `module load singularity-CPEbits`. In addition to setting these bind mounts, one must also explicitly set the environment variable `NCCL_SOCKET_IFNAME=hsn` to workaround RCCL not being able to automatically identify the correct network interfaces to use.
- To workaround a problem with the accessing a single MIOpen SQLite database (used by ROCm) on the Lustre file systems from multiple nodes, one must set the environment variables `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to a node local location, e.g. a folder in /tmp.

This entire setup is shown in `train_multi_node_ddp_env_setup.sh` which launches the training on 4 nodes in standard-g for a total of 4 x 8 = 32 GCDs used in the training.

### NOTE: Checking correct load of aws-ofi-rccl plugin

To check if the aws-ofi-rccl is being used in the training, you may set the following NCCL/RCCL debug environment variables and inspect the standard output from the training.

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
```

The standard output includes lines like `NCCL INFO NET/OFI Using aws-ofi-rccl 1.4.0` if the aws-ofi-rccl plugin has been loaded correctly.

## Other workarounds that may be needed

A few other workaround may - or may not - be needed for the training on LUMI-G nodes to succeed:

- It may be necessary to "warm start" the GPUs before running the training, e.g by calling rocm-smi.
- It may be necessary to set/unset the `CXI_FORK_SAFE/CXI_FORK_SAFE_HP` environment variables when using multiple PyTorch `Dataloader` workers. Both of these are set to `1` in the LUMI ROCm container base image, which may lead to crashes if specifying `workers` > 0 as an argument to PyTorch `Dataloader`s.

# Reference solutions to the hands-on exercises for 05 Running containers on LUMI

## Exercise 1

> 1. Select the following PyTorch container `/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif` found on LUMI.
> 2. Run the `Hello_LUMI_GPU_World.py` Python script inside the container on:
>    - A LUMI login node
>    - A LUMI-G compute node

To run the `Hello_LUMI_GPU_World.py` Python script using one of the LAIF PyTorch container, we must remember to:

1. Bind mount the folder in which the `Hello_LUMI_GPU_World.py` script is placed. If the script is placed in the `/project` or `/scratch` you can just load the correct bindings. 
2. Run the container using `singularity exec`
3. Activate the conda environment in the container by running `$WITH_CONDA` in the container
4. Submit the job using `srun` when using a LUMI-G compute node

On a LUMI login node, it may be done by:

```bash
$ module load Local-LAIF lumi-aif-singularity-bindings
$ singularity exec /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif bash -c "python3 Hello_LUMI_GPU_World.py"
Hello LUMI GPU World from uan02
********************************************************************************
 - We are running in the Singularity container /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif
 - We are running Python version 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0] from /opt/venv/bin/python3
 - The number of GPUs (really GCDs) available to PyTorch is 0
 - Our SLURM job ID is N/A
********************************************************************************

$
```

On a LUMI-G node, it may be done by:

```bash
$ module load Local-LAIF lumi-aif-singularity-bindings
$ srun --account=project_465002757 --partition=small-g --time=00:00:30 --nodes=1 --gpus=4 singularity exec /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif bash -c "python3 Hello_LUMI_GPU_World.py"

srun: job 19023213 queued and waiting for resources
srun: job 19023213 has been allocated resources
Hello LUMI GPU World from nid007898
********************************************************************************
 - We are running in the Singularity container /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif
 - We are running Python version 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0] from /opt/venv/bin/python3
 - The number of GPUs (really GCDs) available to PyTorch is 4
 - Our SLURM job ID is 19023213
********************************************************************************
```

> [!IMPORTANT]
> The number of GPUs/GCDs available to PyTorch is based on how many you request from SLURM. The default is 0!

> [!NOTE]

> It is a good idea to copy the `lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif` container to your project folder and run it from there to enable you to reproduce your results. Containers may be removed in the future!

## Exercise 2

> 1. Pick a container from [Docker Hub](https://hub.docker.com/), e.g. [the official Alpine Docker image](https://hub.docker.com/_/alpine), and pull it to LUMI using Singularity.
>     - Make sure the Singularity cache is not filling up your home folder (hint: see the [LUMI Docs  container page](https://docs.lumi-supercomputer.eu/software/containers/singularity/#pulling-container-images-from-a-registry))
>     - Once Singularity has created the SIF file, you can use it like any other container on LUMI.

To pull containers from Docker Hub without filling up our home folder with Singularity temp/cache files, we must remember to:

1. Use the `docker://<user>/<image>:<tag>` URI specifier with `singularity pull`
2. Set `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` environment variables to another location than our home folder

Pulling version/tag 3.23.4 of the alpine container on a LUMI login node may be done by:

```bash
$ export SINGULARITY_TMPDIR=/tmp/$USER
$ export SINGULARITY_CACHEDIR=/tmp/$USER
$ singularity pull docker://alpine:3.23.4
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
3.3MiB / 3.3MiB [===========================================================] 100 % 24.5 KiB/s 0s
INFO:    Extracting OCI image...
INFO:    Inserting Singularity configuration...
INFO:    Creating SIF file...
$
$ ls -al alpine_3.23.4.sif
-rwxrwx--- 1 username project_465002757 3837952 Jun  3 16:17 alpine_3.23.4.sif
$
```

which generates the `alpine_3.23.4.sif` container.

> [!IMPORTANT]
> There is no automatic cleaning of `/tmp` on the LUMI login nodes. You have to delete the Singularity temp/cache files under `/tmp/$USER` yourself when you are done pull/building containers!

## Exercise 3

> 1. Find a LAIF container that has `deepspeed` pre-installed.
> 2. Open an interactive Python interpreter in the interactive container shell and `import deepspeed`. 

To find a container with deepspeed pre-installed, we have different options: 

1. You can look at the documentation [here](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/) or checkout the release documentation and build scripts [here](https://github.com/lumi-ai-factory/laifs-container-recipes). 
2. Furthermore, you can directly `pip list` the available packages for a container with:
   ```bash
   export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
   singularity run $SIF pip list 
   ```
   This will output:
   ```bash
   Package                                  Version
    ---------------------------------------- --------------------------------------
    absl-py                                  2.4.0
    accelerate                               1.13.0
    aiohappyeyeballs                         2.6.1
    aiohttp                                  3.13.5
    aiosignal                                1.4.0
    alembic                                  1.18.4
    amd-quark                                0.11.1
    amdsmi                                   26.0.2+39589fda
    aniso8601                                10.0.1
    annotated-doc                            0.0.4
    annotated-types                          0.7.0
    anthropic                                0.95.0
    anyio                                    4.13.0
    apex                                     1.10.0+lumi.aif.gfx90a.73423b4
    argon2-cffi                              25.1.0
    argon2-cffi-bindings                     25.1.0
    arrow                                    1.4.0
    astor                                    0.8.1
    asttokens                                3.0.1
    async-lru                                2.3.0
    attrs                                    26.1.0
    auto-round                               0.12.2
    azure-core                               1.39.0
    azure-identity                           1.25.3
    azure-storage-blob                       12.28.0
    babel                                    2.18.0
    beautifulsoup4                           4.14.3
    bitsandbytes                             0.49.2
    blake3                                   1.0.8
    bleach                                   6.3.0
    blinker                                  1.9.0
    boto3                                    1.42.89
    botocore                                 1.42.89
    build                                    1.4.3
    cachetools                               7.0.5
    causal_conv1d                            1.6.1+lumi.aif.gfx90a.0d2252d
    cbor2                                    5.9.0
    certifi                                  2026.2.25
    cffi                                     2.0.0
    charset-normalizer                       3.4.7
    click                                    8.3.2
    cloudpickle                              3.1.2
    colorama                                 0.4.6
    comm                                     0.2.3
    compressed-tensors                       0.15.1a20260414
    conch-triton-kernels                     1.2.1
    contourpy                                1.3.3
    cryptography                             46.0.7
    cxxfilt                                  0.3.0
    cycler                                   0.12.1
    databricks-sdk                           0.102.0
    datasets                                 4.8.4
    debugpy                                  1.8.20
    decorator                                5.2.1
    deepspeed                                0.18.8+lumi.aif.dsops
    .....
   ```
 
To successfully `import deepspeed` in the container, we simply :

1. Open an interactive shell in the container using `singularity shell`

On a LUMI login node, it may be done by:

```bash
$ module load Local-LAIF lumi-aif-singularity-bindings
$ singularity shell /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
Singularity> python
Python 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import deepspeed
Exception caught: rsmi_init.
/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:971: UserWarning: Can't initialize amdsmi - Error code: 34
  raw_cnt = _raw_device_count_amdsmi()
[2026-06-04 12:27:32,806] [WARNING] [real_accelerator.py:183:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
/opt/venv/lib/python3.12/site-packages/apex/transformer/functional/fused_rope.py:54: UserWarning: Using the native apex kernel for RoPE.
  warnings.warn("Using the native apex kernel for RoPE.", UserWarning)
>>> exit()
Singularity> exit
exit
```

The warnings are expected because we run the shell on the login node without GPUs. 

Note: If you do not load the `lumi-aif-singularity-bindings` module you may miss crucial folders and files for slurm to run correctly on the compute nodes. 

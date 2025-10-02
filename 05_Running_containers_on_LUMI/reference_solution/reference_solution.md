# Reference solutions to the hands-on exercises for 05 Running containers on LUMI

## Exercise 1

> 1. Select one of the PyTorch containers found in /appl/local/containers/sif-images/ on LUMI.
> 2. Run the `Hello_LUMI_GPU_World.py` Python script inside the container on:
>    - A LUMI login node
>    - A LUMI-G compute node

For this exercise, we may use e.g. the lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif container found in /appl/local/containers/sif-images/.

To run the `Hello_LUMI_GPU_World.py` Python script using one of the LUMI PyTorch container, we must remember to:

1. Bind mount the folder in which the `Hello_LUMI_GPU_World.py` script is placed (if not it's not your home folder)
2. Run the container using `singularity exec`
3. Activate the conda environment in the container by running `$WITH_CONDA` in the container
4. Submit the job using `srun` when using a LUMI-G compute node

On a LUMI login node, it may be done by:

```bash
$ module use /appl/local/containers/ai-modules
$ module load singularity-AI-bindings
$ singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif bash -c "\$WITH_CONDA; python3 Hello_LUMI_GPU_World.py"
Hello LUMI GPU World from uan03
********************************************************************************
 - We are running in the Singularity container /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
 - We are running Python version 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] from /opt/miniconda3/envs/pytorch/bin/python3
/opt/miniconda3/envs/pytorch/lib/python3.12/site-packages/torch/cuda/__init__.py:721: UserWarning: Can't initialize amdsmi - Error code: 34
  warnings.warn(f"Can't initialize amdsmi - Error code: {e.err_code}")
 - The number of GPUs (really GCDs) available to PyTorch is 0
 - Our SLURM job ID is N/A
********************************************************************************
$
```

On a LUMI-G node, it may be done by:

```bash
$ module use /appl/local/containers/ai-modules
$ module load singularity-AI-bindings
$ srun --account=project_465002178 --partition=small-g --time=00:00:30 --nodes=1 --gpus=4 singularity exec /project/project_465002178/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif bash -c "\$WITH_CONDA; python3 Hello_LUMI_GPU_World.py"

srun: job 11170342 queued and waiting for resources
srun: job 11170342 has been allocated resources
Hello LUMI GPU World from nid007856
********************************************************************************
 - We are running in the Singularity container /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
 - We are running Python version 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] from /opt/miniconda3/envs/pytorch/bin/python3
 - The number of GPUs (really GCDs) available to PyTorch is 4
 - Our SLURM job ID is 11170342
********************************************************************************
$
```

> [!IMPORTANT]
> The number of GPUs/GCDs available to PyTorch is based on how many you request from SLURM. The default is 0!

> [!NOTE]
> It is a good idea to copy the `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif` container to your project folder and run it from there to enable you to reproduce your results. We may remove or replace `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif` at any point in time!

## Exercise 2

> 1. Pick a container from [Docker Hub](https://hub.docker.com/), e.g. [the official Alpine Docker image](https://hub.docker.com/_/alpine), and pull it to LUMI using Singularity.
>     - Make sure the Singularity cache is not filling up your home folder (hint: see the [LUMI Docs  container page](https://docs.lumi-supercomputer.eu/software/containers/singularity/#pulling-container-images-from-a-registry))
>     - Once Singularity has created the SIF file, you can use it like any other container on LUMI.

To pull containers from Docker Hub without filling up our home folder with Singularity temp/cache files, we must remember to:

1. Use the `docker://<user>/<image>:<tag>` URI specifier with `singularity pull`
2. Set `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` environment variables to another location than our home folder

Pulling version/tag 3.19.1 of the alpine container on a LUMI login node may be done by:

```bash
$ export SINGULARITY_TMPDIR=/tmp/$USER
$ export SINGULARITY_CACHEDIR=/tmp/$USER
$ singularity pull docker://alpine:3.19.1
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
3.3MiB / 3.3MiB [===========================================================] 100 % 24.5 KiB/s 0s
INFO:    Extracting OCI image...
INFO:    Inserting Singularity configuration...
INFO:    Creating SIF file...
$
$ ls -al alpine_3.19.1.sif
-rwxrwx---  1 javicher pepr_javicher 3379200 Nov 18 00:22 alpine_3.19.1.sif
$
```

which generates the `alpine_3.19.1.sif` container.

> [!IMPORTANT]
> There is no automatic cleaning of `/tmp` on the LUMI login nodes. You have to delete the Singularity temp/cache files under `/tmp/$USER` yourself when you are done pull/building containers!

## Exercise 3

> 1. Open an interactive shell in the LUMI TensorFlow+Horovod container
> 2. Open an interactive Python interpreter in the interactive container shell and (successfully) `import horovod.tensorflow`

To successfully import Horovod+Tensorflow in the container, we must remember to:

1. Open an interactive shell in the container using `singularity shell`
2. Bind mount the CPE bits when opening the container shell, as Horovod uses MPI that requires parts of the CPE from LUMI
3. Activate the conda environment in the container by running `$WITH_CONDA` in the container shell

On a LUMI login node, it may be done by:

```bash
$ singularity shell --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4   /appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-11-25 23:13:07.472850: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: FFT
2024-11-25 23:13:09.704705: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-25 23:13:11.553455: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: DNN
>>> exit()
(tensorflow) Singularity> exit
exit
```

Remember that instead of manually specifying the bind mounts, you may load the `singularity-AI-bindings` module:

```bash
$ module use /appl/local/containers/ai-modules
$ module load singularity-AI-bindings
$ singularity shell /appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-11-25 23:27:34.216421: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: FFT
2024-11-25 23:27:46.333772: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-25 23:27:52.266194: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: DNN
>>> exit()
(tensorflow) Singularity> exit
exit
```

If you don't bind mount the CPE bits, you will get an error about `libmpi_cray.so.12` not being available:

```bash
$ singularity shell /appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-11-18 00:30:03.593024: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: FFT
2024-11-18 00:30:05.430617: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-18 00:30:07.082782: E external/local_xla/xla/stream_executor/plugin_registry.cc:91] Invalid plugin kind specified: DNN
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/__init__.py", line 27, in <module>
    from horovod.tensorflow import elastic
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/elastic.py", line 24, in <module>
    from horovod.tensorflow.functions import broadcast_object, broadcast_object_fn, broadcast_variables
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/functions.py", line 24, in <module>
    from horovod.tensorflow.mpi_ops import allgather, broadcast, broadcast_
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/mpi_ops.py", line 53, in <module>
    raise e
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/mpi_ops.py", line 50, in <module>
    MPI_LIB = _load_library('mpi_lib' + get_ext_suffix())
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/horovod/tensorflow/mpi_ops.py", line 45, in _load_library
    library = load_library.load_op_library(filename)
  File "/opt/miniconda3/envs/tensorflow/lib/python3.10/site-packages/tensorflow/python/framework/load_library.py", line 54, in load_op_library
    lib_handle = py_tf.TF_LoadLibrary(library_filename)
tensorflow.python.framework.errors_impl.NotFoundError: libmpi_cray.so.12: cannot open shared object file: No such file or directory
>>> exit()
Singularity> exit
```

If you don't activate the conda environment, it will use the container default Python, which does not have TensorFlow and Horovod installed:

```bash
$ singularity shell --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4   /appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif
Singularity> python3
Python 3.6.15 (default, Sep 23 2021, 15:41:43) [GCC] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'horovod'
>>> exit()
Singularity> exit
```

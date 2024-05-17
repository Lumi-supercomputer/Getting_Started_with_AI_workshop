# Suggested solutions to the hands-on exercises for 05 Running containers on LUMI

## Exercise 1

> Run the `Hello_LUMI_GPU_World.py` Python script using one of the LUMI PyTorch containers on:
> * A LUMI login node
> * A LUMI-G compute node

To run the `Hello_LUMI_GPU_World.py` Python script using one of the LUMI PyTorch container, we must remember to:

1. Run the container using `singularity exec`
2. Activate the conda environment in the container by running `$WITH_CONDA` in the container
3. Submit the job using `srun` when using a LUMI-G compute node

On a LUMI login node, it may be done by:

```bash
$ singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif bash -c "\$WITH_CONDA; python3 Hello_LUMI_GPU_World.py"
Hello LUMI GPU World from uan01
********************************************************************************
 - We are running in the Singularity container /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif
 - We are running Python version 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] from /opt/miniconda3/envs/pytorch/bin/python3
 - The number of GPUs (really GCDs) available to PyTorch is 0
 - Our SLURM job ID is N/A
********************************************************************************    
```

On a LUMI-G node, it may be done by:

```bash
$ srun --account=project_465001063 --partition=small-g --nodes=1 --gpus=4 singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif bash -c "\$WITH_CONDA; python3 Hello_LUMI_GPU_World.py"
srun: job 7002601 queued and waiting for resources
srun: job 7002601 has been allocated resources
Hello LUMI GPU World from nid005027
********************************************************************************
 - We are running in the Singularity container /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif
 - We are running Python version 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] from /opt/miniconda3/envs/pytorch/bin/python3
 - The number of GPUs (really GCDs) available to PyTorch is 4
 - Our SLURM job ID is 7002601
********************************************************************************
```

Note that the number of GPUs/GCDs available to PyTorch is based on how many you request from SLURM. The default is 0!

**Remember that it is a good idea to copy the `lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif` container to your project folder and run it from there to enable you to reproduce your results. We may remove or replace `lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif` at any point in time!**

## Exercise 2

> Pick a container from [Docker Hub](https://hub.docker.com/), e.g. [the official Alpine Docker image](https://hub.docker.com/_/alpine), and pull it to LUMI using Singularity. Make sure the Singularity cache is not filling up your home folder (hint: see https://docs.lumi-supercomputer.eu/software/containers/singularity/#pulling-container-images-from-a-registry)

To pull containers from Docker Hub without filling up our home folder with Singularity temp/cache files, we must remember to:

1. Use the `docker://<user>/<image>:<tag>` URI specifier with `singularity pull`
2. Set `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` environment variables to another location than our home folder

Pulling version/tag 3.19.1 of the alpine container on a LUMI login node may be done by:

```bash
$ export SINGULARITY_TMPDIR=/tmp/$USER
$ export SINGULARITY_CACHEDIR=/tmp/$USER
$ singularity pull docker://alpine:3.19.1
INFO:    Converting OCI blobs to SIF format
WARNING: 'nodev' mount option set on /tmp, it could be a source of failure during build process
INFO:    Starting build...
Getting image source signatures
Copying blob 4abcf2066143 done  
Copying config bc4e4f7999 done  
Writing manifest to image destination
Storing signatures
2024/05/03 10:17:08  info unpack layer: sha256:4abcf20661432fb2d719aaf90656f55c287f8ca915dc1c92ec14ff61e67fbaf8
2024/05/03 10:17:08  warn xattr{etc/shadow} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2024/05/03 10:17:08  warn xattr{/tmp/schouoxv/build-temp-1595731751/rootfs/etc/shadow} destination filesystem does not support xattrs, further warnings will be suppressed
INFO:    Creating SIF file...
```

which generates the `alpine_3.19.1.sif` container.

**Remember that there is no automatic cleaning of `/tmp` on the LUMI login nodes. You have to delete the Singularity temp/cache files under `/tmp/$USER` yourself when you are done pull/building containers!**

## Exercise 3

> Open an interactive Python interpreter in the LUMI TensorFlow+Horovod container and (successfully) `import horovod.tensorflow`

To successfully import Horovod+Tensorflow in the container, we must remember to:

1. Open an interactive shell in the container using `singularity shell`
2. Bind mount the CPE bits when opening the container shell, as Horovod uses MPI that requires parts of the CPE from LUMI
3. Activate the conda environment in the container by running `$WITH_CONDA` in the container shell

On a LUMI login node, it may be done by:

```bash
$ singularity shell --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 /appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif 
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-05-03 10:32:19.010478: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

Remember that instead of manually specifying the bind mounts, you may load the `singulariy-CPE-bits` module:

```bash
$ module use /project/project_465001063/modules
$ module load singularity-CPEbits
$ singularity shell /appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif 
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-05-03 10:38:05.846689: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

If you don't bind mount the CPE bits, you will get an error about `libmpi_cray.so.12` not being available:

```bash
£ singularity shell /appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif 
Singularity> $WITH_CONDA
(tensorflow) Singularity> python3
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
2024-05-03 10:34:42.720330: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
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
```

If you don't activate the conda environment, it will use the container default Python, which does not have TensorFlow and Horovod installed:

```bash
$ singularity shell /appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif 
Singularity> python3
Python 3.6.15 (default, Sep 23 2021, 15:41:43) [GCC] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod.tensorflow
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'horovod'
```

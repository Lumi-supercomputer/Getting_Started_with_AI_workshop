# 07 Virtual environments to iterate and test


This is a short overview of how to extend the containers used in lecture 06 with additional python packages via virtual environments. This approach can be useful for developing and testing as it doesn't require rebuilding a container from scratch every time a new package is added.

> [!WARNING]
> This should not be the default way of installing python packages as it puts a lot of strain on the Lustre file system. Once you have a complete set of python packages and their versions, always create a new container.

## Requirements

For this exercise we assume you have built a container from a conda environment file via something like:
```bash
module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif --conda-env=minimal_pytorch.yml
```

## Set up virtual environment

First we run a shell inside the container
```bash
singularity shell --bind /pfs,/scratch,/projappl,/project,/flash,/appl minimal_pytorch.sif
```
Note that setting `--bind` is optional, if you want to keep the virtual environment files in your current folder it is however necessary. Alternatively, you achieve the same by
```bash
module use /project/project_465001063/modules
module load singularity-userfilesystems
singularity shell minimal_pytorch.sif
```

First, let us take a look at all installed python packages in the container via `pip list`
```bash
Singularity> pip list
Package             Version
------------------- --------------
colorama            0.4.6
filelock            3.13.4
fsspec              2024.3.1
gmpy2               2.1.5
Jinja2              3.1.3
lightning-utilities 0.11.2
MarkupSafe          2.1.5
mpmath              1.3.0
networkx            3.3
numpy               1.26.4
packaging           24.0
pillow              10.3.0
pip                 24.0
pretty-errors       1.2.25
pytorch-triton-rocm 2.2.0
setuptools          69.5.1
sympy               1.12
torch               2.2.2+rocm5.6
torchaudio          2.2.2+rocm5.6
torchvision         0.17.2+rocm5.6
typing_extensions   4.11.0
wheel               0.43.0
```
In order to install additional packages we create a virtual environment via `venv` and activate it
```bash
python -m venv myenv --system-site-packages
source myenv/bin/activate
```
The `--system-site-packages` flag gives the virtual environment access to the system site packages.
After activating the virtual environment we can now install custom packages via pip, for example:
```bash
pip install torchmetrics
```
The new package will then be available along side the packages in the container
```bash 
(myenv) Singularity> pip list
Package             Version
------------------- --------------
colorama            0.4.6
filelock            3.13.4
fsspec              2024.3.1
gmpy2               2.1.5
Jinja2              3.1.3
lightning-utilities 0.11.2
MarkupSafe          2.1.5
mpmath              1.3.0
networkx            3.3
numpy               1.26.4
packaging           24.0
pillow              10.3.0
pip                 24.0
pretty-errors       1.2.25
pytorch-triton-rocm 2.2.0
setuptools          65.5.0
sympy               1.12
torch               2.2.2+rocm5.6
torchaudio          2.2.2+rocm5.6
torchmetrics        1.4.0
torchvision         0.17.2+rocm5.6
typing_extensions   4.11.0
wheel               0.43.0
```
We can check the location of the installed packages via
```Python
(myenv) Singularity> python
Python 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os
>>> import torchvision
>>> import torchmetrics
>>> os.path.abspath(torchvision.__file__)
'/opt/conda/envs/conda_container_env/lib/python3.11/site-packages/torchvision/__init__.py'
>>> os.path.abspath(torchmetrics.__file__)
'/pfs/lustrep4/projappl/project_465001063/decristo/contaienr_virt_env/myenv/lib/python3.11/site-packages/torchmetrics/__init__.py'
```
As we can see, the new package is installed in our virtual environment whereas the other packages are installed in the container.

Quick side node: This `venv` approach may also be used with the LUMI application containers, e.g. `/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif`. For these containers it is required to activating the `conda` environment (`$WITH_CONDA`) before creating the `venv`. Also, building a (final) container from e.g. `/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif` + a `venv` is not directly supported by `cotainr`.

## Cleaning up
After having found all packages needed for our purpose, we should create a new container with an updated conda environment file. The virtual environment should then be deleted
```bash
rm -rf myenv
```

## Exercise
Extend one of your existing containers with a python package of your choice following this approach.

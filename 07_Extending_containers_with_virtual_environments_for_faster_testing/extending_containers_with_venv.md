# Extending containers with virtual environments for faster testing


This is a short example of how to extend the containers built via `cotainr` via virtual environments. This approach can be useful for developing and testing as it doesn't require rebuilding a container from scratch every time a new package is added.

> [!WARNING]
> This should not be the default way of installing python packages as it puts a lot of strain on the Lustre file system. Once you have a complete set of python packages and their versions, always create a new container.

## Requirements

We assume you have built a container from a `conda` environment file via something like:
```bash
module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif --conda-env=minimal_pytorch.yml
```

## Set up virtual environment

First we run a shell inside the container
```bash
singularity shell --bind /pfs,/scratch,/projappl,/project,/flash,/appl minimal_pytorch.sif
```
Note that setting `--bind` is optional, you achieve the same by
```bash
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems
singularity shell minimal_pytorch.sif
```

In order to install additional packages we create a virtual environment via `venv` and activate it inside the container
```bash
python -m venv myenv --system-site-packages
source myenv/bin/activate
```
The `--system-site-packages` flag gives the virtual environment access to the packages from the container.

## Install custom packages

After activating the virtual environment we can now install custom packages via pip, for example:
```bash
pip install torchmetrics
```

## Run container with `venv` packages
If we want to run the container with the freshly installed packages in a batch script, we need to first source the `venv` before executing the python script:
```bash
singularity exec $CONTAINER bash -c "source myenv/bin/activate && python my_script.py"
```

## Cleaning up
After having found all packages needed for our project, we should create a new container with an updated `conda` environment file. The virtual environment should then be deleted
```bash
rm -rf myenv
```

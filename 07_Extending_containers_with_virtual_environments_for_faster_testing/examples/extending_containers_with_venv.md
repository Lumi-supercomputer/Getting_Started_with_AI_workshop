# Extending containers with virtual environments for faster testing

This is a short example of how to extend the containers built via `cotainr` via virtual environments. This approach can be useful for developing and testing as it doesn't require rebuilding a container from scratch every time a new package is added.

## Requirements

We assume you have built a container from a `conda` environment file via something like:
```bash
module load LUMI/24.03 cotainr   
cotainr build minimal_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.0.3.sif --conda-env=minimal_pytorch.yml --accept-license
```

## Set up a virtual environment

First, we run a shell inside the container
```bash
singularity shell --bind /pfs,/scratch,/projappl,/project,/flash,/appl minimal_pytorch.sif
```
Note that setting `--bind` is optional, you achieve the same by
```bash
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings
singularity shell minimal_pytorch.sif
```

In order to install additional packages, we create a virtual environment via `venv` and activate it inside the container
```bash
python -m venv myenv --system-site-packages
source myenv/bin/activate
```
The `--system-site-packages` flag gives the virtual environment access to the packages from the container.

## Install custom packages

After activating the virtual environment, we can now install custom packages via pip, for example:
```bash
pip install torchmetrics
```

## Run container with `venv` packages
If we want to run the container with the freshly installed packages in a batch script, we need to first source the `venv` before executing the python script:
```bash
singularity exec minimal_pytorch.sif bash -c "source myenv/bin/activate && python my_script.py"
```

> [!WARNING]
> You should not stop here, as this way of installing python packages creates typically thousands of small files. This puts a lot of strain on the Lustre file system and might exceed your file quota. Choose one of the following options next:


## Option 1: Create a new container with `cotainr`
After having found all packages needed for our project, we should create a new container with an updated `conda` environment file. The virtual environment should then be deleted
```bash
cotainr build updated_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-6.0.3.sif --conda-env=updated_pytorch.yml --accept-license
rm -rf myenv
```


## Option 2: Turn `myenv` into a SquashFS file
Alternatively, we can turn the `myenv` directory into a SquashFS file and bind mount it to the container:
```bash
mksquashfs myenv myenv.sqsh
rm -rf myenv
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
singularity exec -B myenv.sqsh:/user-software:image-src=/ minimal_pytorch.sif python my_script.py
```

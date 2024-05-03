# 05 Containers on LUMI

## Examples

* **Building a container using SingularityCE+proot**: Use the `build_ubuntu_tree.sh` script to build the container `ubuntu_tree.sif` defined in the Singularity definition file `ubuntu_tree.def`. This new container is simply the latest Ubuntu 22.04 Docker Hub image with the `tree` package installed using the Ubuntu Apt package manager. Note that you only need to load the `systools` module on LUMI to have Singularity automatically pick up `proot` from it to do an unpriviledged build on LUMI, i.e. no need for `root`, `sudo`, `--fakeroot`, etc.
* **Showing the top directory trees on LUMI and in an Ubuntu container**: Run the `print_directory_trees.sh` script on LUMI to print:
   1. The `tree -L 1 /` of the `ubuntu_tree.sif` container
   2. The `tree -L 1 /` of LUMI
   3. The `tree -L 1 /` of `ubuntu_tree.sif` with the `/project/project_465001063` folder from LUMI bind mounted

## Hands-on exercises

1. Run the `Hello_LUMI_GPU_World.py` Python script using the LUMI PyTorch container on:
   * A LUMI login node
   * A LUMI-G compute node

2. Pick a container from [Docker Hub](https://hub.docker.com/) and pull it to LUMI using Singularity. Make sure the Singularity cache is not filling up your home folder (hint: see https://docs.lumi-supercomputer.eu/software/containers/singularity/#pulling-container-images-from-a-registry)

3. Open an interactive Python interpreter in the LUMI TensorFlow+Horovod container and (successfully) `import horovod.tensorflow`.

Suggested solutions to the exercises are given in `05_exercise_solutions.md`.

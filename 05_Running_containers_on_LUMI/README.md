# 05 Running containers on LUMI

## Examples

- **Building a container using SingularityCE+proot**: The [build_ubuntu_tree.sh](examples/build_ubuntu_tree.sh) script may be used to build the container `ubuntu_tree.sif` defined in the Singularity definition file [ubutnu_tree.def](examples/ubuntu_tree.def). This new container is simply the latest Ubuntu 22.04 Docker Hub image with the `tree` package installed using the Ubuntu Apt package manager. Note that you only need to load the `systools` module on LUMI to have Singularity automatically pick up `proot` from it to do an unpriviledged build on LUMI, i.e. no need for `root`, `sudo`, `--fakeroot`, etc.
- **Showing the top directory trees on LUMI and in an Ubuntu container**: The [print_directory_trees.sh](examples/print_directory_trees.sh) script may be used on LUMI to print:
   1. The `tree -L 1 /` of the `ubuntu_tree.sif` container
   2. The `tree -L 1 /` of LUMI
   3. The `tree -L 1 /` of `ubuntu_tree.sif` with the `/project/project_465001958` folder from LUMI bind mounted

## Hands-on exercises

1. Hello (LUMI GPU) world in a container

   In this exercise you get to practice running a Python script inside an official LUMI container on both a LUMI login node and a LUMI-G compute node.

   1. Select one of the PyTorch containers found in /appl/local/containers/sif-images/ on LUMI.
   2. Run the `Hello_LUMI_GPU_World.py` Python script inside the container on:
      - A LUMI login node
      - A LUMI-G compute node

2. Pulling a Docker container and using it on LUMI

   In this exercise you will learn how to pull and run an existing Docker container on LUMI.

   1. Pick a container from [Docker Hub](https://hub.docker.com/), e.g. [the official Alpine Docker image](https://hub.docker.com/_/alpine), and pull it to LUMI using Singularity.
      - Make sure the Singularity cache is not filling up your home folder (hint: see the [LUMI Docs  container page](https://docs.lumi-supercomputer.eu/software/containers/singularity/#pulling-container-images-from-a-registry))
      - Once Singularity has created the SIF file, you can use it like any other container on LUMI.

3. Correctly running the official LUMI containers

   In this exercise you will learn to correctly bind mount the necessary CPE bits from LUMI and activate the conda environment in the official LUMI containers.

   1. Open an interactive shell in the LUMI TensorFlow+Horovod container
   2. Open an interactive Python interpreter in the interactive container shell and (successfully) `import horovod.tensorflow`

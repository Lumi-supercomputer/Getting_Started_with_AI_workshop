# 06 Building containers from Conda/pip environments

## Examples

- An absolute minimal conda environment specification only including Python 3.12 is provided in [python312.yml](examples/python312.yml).
- The minimal conda environment PyTorch recipe for LUMI-G is provided in [minimal_pytorch.yml](examples/minimal_pytorch.yml). This environment file can also be used with the `/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-mpich-u24r70f21m50t210-20260415_130625.sif` base image. If you want to do exercise 3 on your own do not look at this example until after the exercise. 
- A minimal conda environment for Jax is provided in [jax.yml](examples/jax.yml).

## Hands-on exercises

1. The basics of using cotainr to build containers on LUMI

   In this exercise you get to practice building containers both interactively and non-interactively on LUMI using cotainr.

   1. Using the example [python312.yml](examples/python312.yml) conda environment, use cotainr to build a container:
       - Interactively on a login node
       - Non-interactively on a compute node
   2. Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"` on a login node:
       - In the container you built
       - Directly on LUMI

2. Making changes to the software environment in the container

   In this exercise you will learn how to add additional packages to your containerized environment using cotainr.

   1. Using cotainr, update the container you built using the `python312.yml` conda environment to contain a few extra packages of your choice, e.g. pandas and scikit-learn.
   2. Open an interactive Python interpreter in the container and import your newly added packages.

3. Creative pip installs using cotainr

   In this exercise you will learn how to install Python packages in a container using cotainr when no conda package exists for the package.

   1. Create a conda yaml file based on the `python312.yml` and add ROCm 7.0 versions of torch, torchvision, torchaudio and triton-rocm. 
   2. Build a new container image using cotainr on LUMI-C.
   3. Confirm that pytorch has access to the AMD GPUs on LUMI-G.

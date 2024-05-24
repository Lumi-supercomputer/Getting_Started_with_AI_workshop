# 06 Building containers from Conda/pip environments

## Examples

- An example of a complete PandasAI conda environment specification is provided in [PandasAI.yml](examples/PandasAI.yml).
- An absolute minimal conda environment specification only including Python 3.12 is provided in [python312.yml](examples/python312.yml).
- The minimal conda environment PyTorch recipe for LUMI-G is provided in [minimal_pytorch.yml](examples/minimal_pytorch.yml).
- The minimal conda environment TensorFlow recipe for LUMI-G is provided in [minimal_tensorflow.yml](examples/minimal_tensorflow.yml).
- The minimal conda environment JAX/Flax recipe for LUMI-G is provided in [minimal_jaxflax.yml](examples/minimal_jaxflax.yml).

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

   In this exercise you will learn how to install Python packages in a container using cotainr when no conda package or pip wheel exists for the package.

   1. Create a conda environment file for installing [panopticapi](https://github.com/cocodataset/panopticapi)
   2. Use the conda environment file to build a container for LUMI-C using cotainr

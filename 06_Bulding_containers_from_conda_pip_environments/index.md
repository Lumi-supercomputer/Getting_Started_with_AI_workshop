# 06 Building containers from Conda/pip environments

## Examples

* An example of a complete PandasAI conda environment specification is provided in `PandasAI.yml`.
* An absolute minimal conda environment specification only including Python 3.12 is provided in `python312.yml`.
* The minimal conda environment PyTorch recipe for LUMI-G is provided in `minimal_pytorch.yml`.
* The minimal conda environment TensorFlow recipe for LUMI-G is provided in `minimal_tensorflow.yml`.
* The minimal conda environment JAX/Flax recipe for LUMI-G is provided in `minimal_jaxflax.yml`.

## Hands-on exercises

1. Using the provided `python312.yml` conda environment:
   * Use cotainr to build a container (interactively) on a login node
   * Use cotainr to build the same container (non-intactively) on a compute node
   * Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"` on a login node:
     * In the container you build
     * Directly on LUMI
2. Using cotainr, update the container you built using the `python312.yml` conda environment to contain a few extra packages of your choice. Open an interactive Python interpreter in the container and import your newly added packages.
3. Create a conda environment file for installing [panopticapi](https://github.com/cocodataset/panopticapi) and use it to build a container for LUMI-C using cotainr.

Suggested solutions to the exercises are given in `06_exercise_solutions.md`.

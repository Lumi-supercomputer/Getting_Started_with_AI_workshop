# 06 Building containers from Conda/pip environments

## Examples

* An example of a complete PandasAI conda environment specification is provided in `PandasAI.yml`.
* An absolute minimal conda environment specification only including Python 3.12 is provided in `python312.yml`.
* The minimal PyTorch recipe conda environment is provided in `minimal_pytorch.yml`.

## Hands-on exercises

* Using the provided `python312.yml` conda environment:
  * Use cotainr to build a container (interactively) on a login node
  * Use cotainr to build the same container (non-intactively) on a compute node
  * Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"`:
    * In the container you build
    * Directly on LUMI
* Add a few extra packages to the specification in the `python312.yml` file and build a container using cotainr. Open an interactive Python interpreter in the container and import your newly added packages.
* Create a `conda_env.yml` file for installing [panopticapi](https://github.com/cocodataset/panopticapi) and use it to build a container for LUMI-C using cotainr.
* Create a `conda_env.yml` file for one of your existing Python AI environments and use it to build a container for LUMI-G using cotainr. Remember to specify ROCm versions of the packages that need (AMD) GPU acceleration.

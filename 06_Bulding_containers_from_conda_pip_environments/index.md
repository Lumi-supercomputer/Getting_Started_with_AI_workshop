# 06 Building containers from Conda/pip environments

## Examples

[comment]: <> (List your examples from the lecture here and provide the necessary links to scripts, notebooks, etc. to run them on LUMI)

* An example of a complete PandasAI conda environment specification is provided in `PandasAI.yml`

## Hands-on exercises

[comment]: <> (List your hands-on exercises for the lecture here and provide the necessary links to scripts, notebooks, etc. to run them on LUMI)

* Using the provided `python312.yml` conda environment,
  * Use cotainr to build a container (interactively) on a login node
  * Use cotainr to build the same container (non-intactively) on a compute node
    * Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"`:
      * In the container you build
      * Directly on LUMI
* Add a few extra packages to the specification in the `python312.yml` file and build a container using cotainr. Open an interactive Python interpreter in the container and import your newly added packages.
* Create a conda_env.yml file for installing [panopticapi](https://github.com/cocodataset/panopticapi) and use it to build a container for LUMI-C using cotainr.
* Create a conda_env.yml file for one of your existing Python AI environments and use it to build a container for LUMI-G using cotainr. Remember to specify ROCm versions of the packages that need (AMD) GPU acceleration.

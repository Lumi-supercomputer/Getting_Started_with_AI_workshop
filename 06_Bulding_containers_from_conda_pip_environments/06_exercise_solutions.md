# Suggested solutions to the hands-on exercises for 06 Building containers from conda/pip environments

## Exercise 1

```text
Using the provided `python312.yml` conda environment:
* Use cotainr to build a container (interactively) on a login node
* Use cotainr to build the same container (non-intactively) on a compute node
* Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"`:
  * In the container you build
  * Directly on LUMI
```

To build a container using cotainr on LUMI, we must remember to:

1. Load the cotainr module on LUMI
2. Determine the relevant system (LUMI-C/LUMI-G) or, alternatively, pick a suitable base image
3. Run cotainr using `srun` and piping stdout/stderr when building non-interactively on a compute node

Since the `python312.yml` environment only contains Python 3.12, we don't need ROCm or other special system libraries, so using `--system=lumi-c` with cotainr is sufficient for getting a fairly minimal base image.

On a login node, we may build the container interactively by:

```bash
$ module use /project/project_465001063/modules
$ module load LUMI cotainr
$ cotainr build python312.sif --system=lumi-c --conda-env=python312.yml
```

Note that `module use /project/project_465001063/modules` provides the most recent version of cotainr installed in the AI workshop training project. If you don't include this, you get an older version of cotainr installed in the default LUMI software stack.


On a LUMI-C compute node, we may build the container non-interactively by:

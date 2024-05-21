# 09 Hyper-parameter tuning using Ray

## Hands-on exercises

1. In order to run the ray example we need to extend the `pytorch_transformers.sif` container with the pip package `ray[tune]`.

    **Task:** Extend the container via a virtual environment and install `ray[tune]`. You can follow the instructions in [07_Virtual_environments_to_iterate_and_test](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Virtual_environments_to_iterate_and_test/index.md)
    Note, that you need to source the venv in the batch script.

2. Set up job script for using one whole node for ray tune.

    **Task:** Specify slurm parameters in `run.sh` file to use whole node with all available GPUs.
    **Task:** At the end of the file, set up running the file `GPT-neo-ray-tune.py` in the specified container with the correct arguments.

3. Specify computational resources and parameter space in `GPT-neo-ray-tune.py`.

    **Task:** In line 166, pass the correct number of CPUs and GPUs to ray.
    **Task:** In line 171, choose a reasonable parameter space for 'learning_rate'
    **Task:** In line 178, specify the correct number of CPUs and GPUs for the individual trial runs

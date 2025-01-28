# 08 Scaling to multiple GPUs

## Hands-on exercises

1. Adjust the training script to run with torchrun for multiple GCDs on a single node.

   In this exercise you have to make some changes to the Python training script to prepare it for running on multiple GPUs across several processes.

   We will use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to run the training on all GCDs on a full LUMI node. Torchrun creates and manages one process per GCD, each executing our training script, and provides the following environment variables to each process:
   - `WORLD_SIZE`: The total number of processes.
   - `RANK`: The process number from `0` to `WORLD_SIZE-1`.
   - `LOCAL_WORLD_SIZE`: The total number of processes on the same node.
   - `LOCAL_RANK`: The process number within the same node from `0` to `LOCAL_WORLD_SIZE-1`.
   - `MASTER_ADDR`: The URL of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
   - `MASTER_PORT`: The port on the `MASTER_ADDR` the different processes use to communicate.

   In this exercise we will use all the GCDs available we will not use multiple nodes. Therefore `LOCAL_WORLD_SIZE` and `LOCAL_RANK` will be identical to `WORLD_SIZE` and `RANK`.

   Using torchrun, every process sees all GCDs on a node, so we need to make sure that our script selects one GCD it is going to train on according to its local rank.

   On the other hand, the HuggingFace Trainer will automatically take care of setting up data-parallel training for the model when the above environment variables are set by torchrun, so we do not need to handle that part of the setup manually - but we do need to adjust the batch size handled locally in each process.

   Find the training script in [08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py](08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py). It is the same as the one used earlier for training with a single GCD/GPU.

   1. You will need to make the following changes to the script:

      - select the correct PyTorch device
      - adjust the per-device batch size handled per process
      - (optional) limit printing of outputs to a single process

      Places where you need to edit the file have been marked with `<!!! ACTION REQUIRED ... !!!>`.

2. Adjust the slurm batch file.

   Now you need to change the slurm batch file to request multiple GCDs (GPUs) on a single node and use `torchrun` to start a training job that
   parallelises training across the GCDs.

   1. Edit the slurm batch file [08_Scaling_to_multiple_GPUs/run.sh](run.sh) for single-node multi-gpu training using torchrun.

      You should specify at least the following:
      - the correct slurm partition
      - number of GPUs requested (8)
      - number of CPUs requested
      - RAM requested (we recommend using 60GB per requested GPU to leave some room for the OS overhead)
      - requested runtime (20 minutes should be plenty to finish training and running evaluation)

      It can also be helpful to specify a name for the slurm logfile that contains the command line outputs of the script.

      > ** Tip **
      >
      > You can use a different `--model-name` than in Exercise 3, to start a fresh training run without overwriting your
      > earlier results. The environment variable `MODEL_NAME` is a suggestion for a name that you can use.

      You will also need to add the relevant parts for setting up the PyTorch software environment (these are the same as for Exercise `03_Your_first_AI_training_job_on_LUMI`).

      To invoke torchrun from the batch file, follow the [Single-node multi-worker usage example on the torchrun website](https://pytorch.org/docs/stable/elastic/run.html#single-node-multi-worker).

   2. Run your job using `sbatch run.sh`.

   3. Compare how the run time differs between running on a full node and the previous run on a single GCD.

      You don't necessarily need to wait for the run to finish but can compare the estimated total time given by the progress bar.

3. (Optional/Bonus): Set up CPU bindings.

   In order to achieve optimal CPU-GPU data transfer performance we can ensure that each script runs on the CPU cores closest to the respective GPU.
   As we are using torchrun to manage the worker processes, we cannot handle these CPU bindings via slurm but must set them up in our Python training script.

   1. Edit [08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py](GPT-neo-IMDB-finetuning.py) to set up the correct CPU-GPU bindings based on the processes rank.

      You can find a [figure showing which cores are closest to which GCD](https://docs.lumi-supercomputer.eu/assets/images/lumig-cpu-gpu-links.svg) on the [LUMI Docs LUMI-G page](https://docs.lumi-supercomputer.eu/hardware/lumig/).

      > **Tip**
      >
      > Use the `psutil.Process().cpu_affinity(...)` function to set the binding from inside the Python script.

4. (Optional/Bonus): Running without PyTorch.

   We can also start worker processes directly without using torchrun to have direct control over all processes.

   1. Change the slurm batch script to
      - instruct slurm to start the appropriate number of processes,
      - set the environment variables mentioned above manually,
      - replace the `torchrun` invocation with direct `python` commands to run the training script.

      > **Note**
      >
      > You can get the hostname of the node running the rank 0 process using the command
      > ```
      > hostname
      > ```

   In this setting you could then also do the CPU bindings from the slurm batch file instead of Python, to keep the training script free of system specific setup.

## Solutions

The folder `reference_solution/` contains an example solution for this exercise parts 1, 2 and 4. `reference_solution/prints_only_from_single_process` extends this to ensure that `print` statements in the code are run only by a single process. `reference_solution/with_cpu_bindings` shows how CPU bindings can be used both from within Python (when using torchrun) and directly via SLURM (exercise part 3).

## Further Examples

- For a native PyTorch DDP example (without using HuggingFace modules), see [/bonus_material/pytorch_cotainr_container_basics](/bonus_material/pytorch_cotainr_container_basics), specifically
  - for torchrun usage, check
    - [train_multi_gpu_ddp_torchrun.py](/bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_torchrun.py)
    - [train_multi_gpu_ddp_torchrun.sh](/bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_torchrun.sh)
  - for manual process setup, check
    - [train_multi_gpu_ddp_env_setup.py](/bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_env_setup.py)
    - [train_multi_gpu_ddp_env_setup.sh](/bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_env_setup.sh)

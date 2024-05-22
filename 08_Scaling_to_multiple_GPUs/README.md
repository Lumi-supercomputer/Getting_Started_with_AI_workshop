# 08 Scaling to multiple GPUs

## Examples

- For a native PyTorch DDP example (without using HuggingFace modules), see [bonus_material/pytorch_cotainr_container_basics](bonus_material/pytorch_cotainr_container_basics), specifically
  - for torchrun usage, check
    - [train_multi_gpu_ddp_torchrun.py](bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_torchrun.py)
    - [train_multi_gpu_ddp_torchrun.sh](bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_torchrun.sh)
  - for manual process setup, check
    - [train_multi_gpu_ddp_env_setup.py](bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_env_setup.py)
    - [train_multi_gpu_ddp_env_setup.sh](bonus_material/pytorch_cotainr_container_basics/train_multi_gpu_ddp_env_setup.sh)

## Hands-on exercises

1. Adjust the training script to run with torchrun for multiple GCDs on a single node.
   
   Find the training script in [08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py](08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py). It is the same as the one used earlier for training with a single GCD/GPU.

   We will use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) to run the training on all GCDs on a LUMI node. Torchrun creates and manages one process per GCD, each executing our training script, and provides the following environment variables to each process:
   - `WORLD_SIZE`: The total number of processes.
   - `RANK`: The process number from `0` to `WORLD_SIZE-1`.
   - `LOCAL_WORLD_SIZE`: The total number of processes on the same node.
   - `LOCAL_RANK`: The process number within the same node from `0` to `LOCAL_WORLD_SIZE-1`.
   - `MASTER_ADDR`: The FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
   - `MASTER_PORT`: The port on the `MASTER_ADDR` to use.

   Using torchrun, every process sees all GCDs on a node, so we need to make sure that our script selects the GCD it is going to train on according to its local rank.

   However, we are using the HuggingFace Trainer which will automatically take care of setting up data-parallel training for the model if it detects the environment variables set by torchrun, so we do not need to handle that manually.

   **Task**: You will need to make the following changes to the script:
   - select the correct PyTorch device
   - adjust the per-device batch size handled per process
   - limit printing of outputs to a single process
  
   Places where you need to edit the file have been marked with `<!!! ACTION REQUIRED ... !!!>`.
  
2. Adjust the slurm batch file.

   **Task**: Edit the slurm batch file [08_Scaling_to_multiple_GPUs/run.sh](08_Scaling_to_multiple_GPUs/run.sh) for single-node multi-gpu training using torchrun.

   You should specify at least the following:
    - the correct slurm partition
    - number of GPUs requested (8)
    - number of CPUs requested
    - RAM requested (we recommend using 60GB per requested GPU to leave some room for the OS overhead)
    - requested runtime
    - the course reservation

   It can also be helpful to specify a name for the slurm logfile that contains the command line outputs of the script.

   You will also need to add the relevant parts for setting up the PyTorch software environment (these are the same as for Exercise `03_Your_first_AI_training_job_on_LUMI`).

   To invoke torchrun from the batch file, follow the [Single-node multi-worker usage example on the torchrun website](https://pytorch.org/docs/stable/elastic/run.html#single-node-multi-worker).

   **Task**: Run your job using `sbatch run.sh`.

   **Task**: Compare how the run time differs between running on a full node and the previous run on a single GCD.

3. Set up CPU bindings.

   In order to achieve optimal CPU-GPU data transfer performance we need to ensure that each script remains on the CPU cores closest to the respective GPU.
   As we are using torchrun to manage the worker processes, we cannot handle these CPU bindings via slurm but must set them up in our Python training script.

   **Task**: Edit [08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py](08_Scaling_to_multiple_GPUs/GPT-neo-IMDB-finetuning.py) to set up the correct CPU-GPU bindings based on the processes rank.

4. (Optional/Bonus): Running without PyTorch.

   We can also start worker processes directly without using torchrun to have direct control over all processes.
   
   **Task**: Change the slurm batch script to
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

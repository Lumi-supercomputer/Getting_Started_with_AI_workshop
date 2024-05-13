# 03 Your first AI training job on LUMI

## Examples

[comment]: <> (List your examples from the lecture here and provide the necessary links to scripts, notebooks, etc. to run them on LUMI)

* ...

## Hands-on exercises

1. Familiarise yourself with the training script.
   
   Check the training script [03_Your_first_AI_training_job_on_LUMI/GPT-neo-IMDB-finetuning.py](03_Your_first_AI_training_job_on_LUMI/GPT-neo-IMDB-finetuning.py). 
   
   You can open it via
   - any command line editor from a login node shell, either via the LUMI web interface or an SSH connection
   - the Visual Studio Code app in the LUMI web interface (use the `interactive` partition)

2. Create a slurm batch file and start a training run.

    In the same directory as the script you can find [03_Your_first_AI_training_job_on_LUMI/run.sh](03_Your_first_AI_training_job_on_LUMI/run.sh), an incomplete slurm batch job file.
    
    You should specify at least the following:
    - the correct slurm partition
    - number of CPUs requested
    - number of GPUs requested (1)
    - RAM requested
    - requested runtime (recommended: 10 minutes, for sub-exercise 4 below)
  
    It can also be helpful to specify a name for the slurm logfile that contains the command line outputs of the script.

    You will also need to add the relevant parts for setting up the PyTorch software environment.
    
    Fill in the missing pieces and start the training using `sbatch run.sh` from a login node shell (LUMI web interface or SSH).

3. Check your job.

    From a login node shell, use the slurm command `squeue --me` to check that your job is running. You can use the `tail -f` command to check the outputs of the job from the slurm log once the job is running.

    You can also check your active jobs from the LUMI web interface: Navigate to Jobs > Active Jobs.

    > [!NOTE]
    > We will cover more details about checking the status and progress of your job in a later exercise.

4. Modify the script to enable it to continue from a check point.

    The script currently always starts training with the GPT-neo model.
    
    Change it so it can load a checkpoint from a previously interrupted training run and resume training. Check the [documentation about HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) for guidance on this.

    You may want to add additional command line arguments to control this behaviour from the slurm batch script.

    Your earlier training job should by now have timed out without completing the training. Use your modified script to resume training from the last checkpoint.

    If your earlier training job is still running, you can stop it using the `scancel` command.

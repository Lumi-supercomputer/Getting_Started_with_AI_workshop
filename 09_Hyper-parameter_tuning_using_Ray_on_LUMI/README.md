# 09 Hyper-parameter tuning using Ray

## Hands-on exercises

In this exercise, we perform Hyper-parameter tuning for the model used in lecture 03. The python code in this lecture, `GPT-neo-ray-tune.py` is based on a trimmed down version of `GPT-neo-IMDB-finetuning.py` from lecture 03.

1. Set up job script for using one whole node for ray tune.

    * Specify slurm parameters in `run.sh` file to use whole node with all available GPUs.
   
2. Fill in the missing pieces (marked with <!!! ACTION REQUIRED ... !!!>) in `GPT-neo-ray-tune.py`
    
    * Pass the correct number of CPUs and GPUs to ray.init()

    * Choose a reasonable parameter space for `learning_rate`. Note, we want to run 8 trials with different parameters in parallel which should take around 6-7 minutes

    * Specify the correct number of CPUs and GPUs for the trial runs
  
3. Submit the batch job and check whether all GPUs are used:

   * After submitting your batch job, run `srun --overlap --pty --jobid=... bash` followed by `rocm-smi` to check whether all 8 GPUs are used close to 100%

4. Check in the output file whether all trials run successfully. 
    
    * During the run, the output should look similar to the figure below. The statement 0.0/1.0 accelerator_type:AMD-Instinct-MI250X can be safely ignored as it is a known bug. This message appears even when all GPUs are fully utilized correctly.

    ![running output](./images/running_output.png)

    * After the job finished, the output should look similar to

    ![desired output](./images/desired_output.png)

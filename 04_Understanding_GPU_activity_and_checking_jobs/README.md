# 04 Understanding GPU activity & checking jobs

These examples are based on the ROCm container provided to you at:
```
/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif 
```

The examples also assume there is an allocation in place to be used for one or more nodes. That could be accomplished with, e.g.:
```
salloc -p small-g --account=project_465001707 --reservation=AI_workshop --gpus-per-node=2 --ntasks-per-node=1 --cpus-per-task=14 --mem-per-gpu=60G --time=0:30:00
```
This is very similiar to what you have been doing with `sbatch` should you be using a run script with:
```
#SBATCH --account=project_465001707
#SBATCH --reservation=AI_workshop
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00
```
The difference is that it gives you a mechanism to just allocate the nodes without running anything. You can then issue `srun` commands interactively which can be useful to experiment more easily. You are always welcome to transition to use `sbatch` if that is preferred.

<!--
We'll also leverage the configuration for singularity provided by:
```
module purge
module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems singularity-CPEbits
``` 
-->

With the allocation and container set we can do a quick smoke test to make sure Pytorch can detect the GPUs available in a node:
```
srun singularity exec \
  /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif  \
    bash -c '$WITH_CONDA ; \
             python -c "import torch; print(torch.cuda.device_count())"'
```

It should yield `2` given that only two GPUs were requested. Note that each time a node is used for the first time, there is a latency to have the container loaded. Running the command above again on the same allocation should complete faster.

## Hands-on exercise

We will leverage here the same LLM example as before with small adaptations. No extra files are needed. You might be interested in collating the different steps in a batch script or run interactively as presented. 

### 1. Let's recover our LLM example.
Here we'll recover our fine-tunning example for IMDB movie review generation:

```
curl -o GPT-neo-IMDB-finetuning.py -L https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/raw/main/03_Your_first_AI_training_job_on_LUMI/reference_solution/GPT-neo-IMDB-finetuning.py
curl -o util.py -L https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/raw/main/03_Your_first_AI_training_job_on_LUMI/util.py
```

### 2. Spin training work
We can now run our training as:

```
mkdir -p torch-cache hf-cache

srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif \
    bash -c '$WITH_CONDA ; cd /workdir ; \
             HIP_VISIBLE_DEVICES=0 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             python -u /workdir/GPT-neo-IMDB-finetuning.py \
               --model-name gpt-imdb-model \
               --output-path /workdir/train-output \
               --logging-path /workdir/train-logging \
               --num-workers 7'
```

While the training runs, let's discover what is the CPU/GPU activity. Note that we are leveraging an allocation with 2 logical GPUs, so we are limiting visibility with the variable `HIP_VISIBLE_DEVICES`. Given that the actually GPU chip has two GCDs (logical GPUs) is better to try monitor on the actually GPU, and not just half of it.

### 3. Monitoring GPU activity

Monitoring in a separate tab can be done by checking you jobID and connect to the first node of the allocation. E.g.:

* Get jobID - in this case `7100665`:
```
squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           7100665   small-g interact samantao  R    1:03:21      1 nid005021
           ...
```
* Start interactive parallel session:
```
srun --jobid 7100665 --interactive --pty /bin/bash
```
* Use `rocm-smi` to monitor GPU activity:
```
watch -n1 rocm-smi
```
This will give a snapshot of the GPU utilization captured by the driver every second:
```
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK     MCLK     Fan  Perf    PwrCap  VRAM%  GPU%  
0    58.0c  324.0W  1650Mhz  1600Mhz  0%   manual  500.0W   98%   100%  
1    49.0c  N/A     800Mhz   1600Mhz  0%   manual  0.0W      0%   0%    
================================================================================
============================= End of ROCm SMI Log ==============================
```
As expected we only have activity on one GCD but the power metrics are per GPU. Note that these numbers needs to be interpreted. For example, if `GPU%` shows `100%` that does NOT necessarily mean the GPU is being well utilized. A better metric is drawn power `AvgPwr`: oscillating around `500.0W` is an indication there is significant compute activity on the full GPU.

Here we see drawn power to oscillate around `300.0W` while a single GCD is being used, which is an indication that we might be compute bound.

### 4. Activate logging reporting GPU activity

Other ways to understand the activity connected to GPU-enabled libraries is to enable logging messages for these libraries. Here are some examples:

* `AMD_LOG_LEVEL=4` - this captures the HIP runtime activity used to copy data and issue kernels into the GPU. 

* `MIOPEN_ENABLE_LOGGING=1` - this captures API activity for the MIOpen library that provides optimized kernels for AI applications. Your application might not use that though,

So, running the following:
```
srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif \
    bash -c '$WITH_CONDA ;  cd /workdir ; \
             HIP_VISIBLE_DEVICES=0 \
             AMD_LOG_LEVEL=4 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             python -u /workdir/GPT-neo-IMDB-finetuning.py \
               --model-name gpt-imdb-model \
               --output-path /workdir/train-output \
               --logging-path /workdir/train-logging \
               --num-workers 7'
```
would return something like the following for a given kernel and its dispatch configuration:
```
:3:hip_module.cpp           :662 : 117659918626 us: 8088 : [tid:0x14b2015e9700]  hipLaunchKernel ( 0x14b5ec183ed0, {32768,1,1}, {512,1,1}, 0x14b2015e71b0, 0, stream:<null> )                                                       :4:command.cpp              :349 : 117659918630 us: 8088 : [tid:0x14b2015e9700] Command (KernelExecution) enqueued: 0x14b151fe3b00                                                                                                  :3:rocvirtual.cpp           :786 : 117659918634 us: 8088 : [tid:0x14b2015e9700] Arg0:   = val:16777216                                                                                                                              
:3:rocvirtual.cpp           :786 : 117659918636 us: 8088 : [tid:0x14b2015e9700] Arg1:   = val:22689590804480                                                                                                                        :3:rocvirtual.cpp           :2853: 117659918639 us: 8088 : [tid:0x14b2015e9700] ShaderName : _ZN2at6native6legacy18elementwise_kernelILi512ELi1EZNS0_15gpu_kernel_implIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENK
UlvE0_clEvENKUlvE5_clEvEUlfE_EEvS5_RKT_EUliE_EEviT1_                                                                                                                                                                                
:4:rocvirtual.cpp           :891 : 117659918644 us: 8088 : [tid:0x14b2015e9700] HWq=0x14b30ee00000, Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1), setup=3, grid=[16777216, 1, 1], workgroup=[512, 1, 1], privat
e_seg_size=0, group_seg_size=0, kernel_obj=0x14b4a5220000, kernarg_address=0x14b30ec73780, completion_signal=0x0                                                                                                                    
:3:hip_module.cpp           :663 : 117659918649 us: 8088 : [tid:0x14b2015e9700] hipLaunchKernel: Returned hipSuccess :    
```
Try to interpret the different kinds of activity.

### 5. Using a profiler to assess GPU activity.

Another way to check for GPU activity is to use a profiler. There is a GPU profiler included in any ROCm instalation: `ROCprofiler`. This profiler is also available inside the containers, so no extra instalations is required. It has a command-line driver called `rocprof` and you can see the options one can use with:
```
srun -n1 singularity exec \
    -B .:/workdir \
   /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif \
    rocprof --help
```
Given that Pytorch uses the HIP runtime in its implementation, one of the most relevant options is `--hip-trace` to instruct the profiler to collect the HIP runtime activity. Another option that is convinient is `--stats` that generates some statistics on the usage of the GPU. 

Just to allow a quicker completion time, let's focus on just a few training steps. For that just open the file `GPT-neo-IMDB-finetuning.py` and replace:
```
        max_steps=1000,
```
with:
```
        max_steps=10,
```
and place a `import sys ; sys.exit(0)` statement after:
```
    trainer.train(resume_from_checkpoint=args.resume)
```

Now we can just run the profiler by preceding our original command with `rocprof`.

```
srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif \
    bash -c '$WITH_CONDA ; cd /workdir ;  \
             HIP_VISIBLE_DEVICES=0 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             rocprof --hip-trace --stats python -u /workdir/GPT-neo-IMDB-finetuning.py \
               --model-name gpt-imdb-model \
               --output-path /workdir/train-output \
               --logging-path /workdir/train-logging \
               --num-workers 7'
```
This will generate a few files named `results.*`. For example, `results.stats.csv` will provide the stats of the kernels that were executed in the GPU in descending order of combined execution time. These, can sometimes be easier to read if imported into a spreadsheet. 

### 6. Visualizing a profile trace
Other file that might be interesting to look at is `results.json`. This can be loaded into the web app `https://ui.perfetto.dev/v46.0-35b3d9845/#/` and will allow you to visualize the GPU execution. Here is a snapshot of the 10 steps executed:

![image](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/raw/main/04_Understanding_GPU_activity_and_checking_jobs/images/profile.png)

### 7. Using Pytorch profiling infrastructure.

Pytorch already provides profiling infrastruture that captures GPU activity as well as ranges for the CPU activities. It can be loaded with:
```
from torch.profiler import profile, ProfilerActivity
```
Then, you can identify the part of the code to profile, e.g. a given epoch. At the start of that part you can create and start the `profile` object:
```
prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
prof.start()
```
and at the end you can stop and create the profile file to be loaded into Perfetto UI tool mentioned above:
```
prof.stop()
prof.export_chrome_trace("trace.json")
```

Let's get our example:
```
curl -o GPT-neo-IMDB-finetuning-profile.py -L https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/raw/main/03_Your_first_AI_training_job_on_LUMI/reference_solution/GPT-neo-IMDB-finetuning.py
```
Use `max_steps=10` and place the profiler start and end around:
```
trainer.train(resume_from_checkpoint=args.resume)
```
Run as before:
```
srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif \
    bash -c '$WITH_CONDA ; cd /workdir ;  \
             HIP_VISIBLE_DEVICES=0 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             python -u /workdir/GPT-neo-IMDB-finetuning-profile.py \
               --model-name gpt-imdb-model \
               --output-path /workdir/train-output \
               --logging-path /workdir/train-logging \
               --num-workers 7'
```
Then you can visualize the file `trace.json`.

A solution `GPT-neo-IMDB-finetuning-profile.py` is available [here](reference_solution/GPT-neo-IMDB-finetuning-profile.py).

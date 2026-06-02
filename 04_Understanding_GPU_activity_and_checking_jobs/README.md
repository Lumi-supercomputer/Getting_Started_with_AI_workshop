# 04 Understanding GPU activity & checking jobs

These examples are based on the ROCm container provided to you at:
```
/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
```

To avoid running into any storage issues, we recomment running the examples from a folder you create in the scratch file system, e.g.:
```
mkdir -p /scratch/project_465002757/$(whoami)
cd /scratch/project_465002757/$(whoami)
```

The examples also assume there is an allocation in place to be used for one or more nodes. That could be accomplished with, e.g.:
```
salloc -p small-g --account=project_465002757 --reservation=AI_workshop_Day1 --gpus-per-node=2 --ntasks-per-node=1 --cpus-per-task=14 --mem-per-gpu=60G --time=0:30:00
```
This is very similiar to what you have been doing with `sbatch` should you be using a run script with:
```
#SBATCH --account=project_465002757
#SBATCH --reservation=AI_workshop_Day1
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00
```
The difference is that it gives you a mechanism to just allocate the nodes without running anything. You can then issue `srun` commands interactively which can be useful to experiment more easily. You are always welcome to transition to use `sbatch` if that is preferred.

With the allocation and container set we can do a quick smoke test to make sure Pytorch can detect the GPUs available in a node:
```
srun singularity exec \
  /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'python -c "import torch; print(torch.cuda.device_count())"'
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
    /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'cd /workdir ; \
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
srun --jobid 7100665 --overlap --pty /bin/bash
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

Here we see drawn power to oscillate around `300.0-400.0W` while a single GCD is being used, which is an indication that we might be compute bound.

An alternative to `rocm-smi` and recommended for the very latest ROCm versions is to use `amd-smi`. Similar information can be obtained with `amd-smi monitor --watch 1`. It is formatted differently thought.

`amd-smi` has also Python interface and it is possible to programatically query GPU activity from your Python code.

### 4. Activate logging reporting GPU activity

Other ways to understand the activity connected to GPU-enabled libraries is to enable logging messages for these libraries. Here are some examples:

* `AMD_LOG_LEVEL=4` - this captures the HIP runtime activity used to copy data and issue kernels into the GPU. 

* `MIOPEN_ENABLE_LOGGING=1` - this captures API activity for the MIOpen library that provides optimized kernels for AI applications. Your application might not use that though,

So, running the following:
```
srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'cd /workdir ; \
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
:3:hip_module.cpp           :812 : 9529939547052 us: [pid:31271 tid: 0x145b173466c0]  hipLaunchKernel ( 0x145bd3dc0e30, {65536,1,1}, {256,1,1}, 0x145b17344480, 0, stream:<null> ) 
:4:hip_device.cpp           :35  : 9529939547073 us: [pid:31271 tid: 0x145b173466c0] NullStream 0x101fe7b0, wait 1  
:4:command.cpp              :169 : 9529939547067 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x1449648f6fe0 complete      
:4:command.cpp              :169 : 9529939547081 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x144964976700 complete                                          
:4:command.cpp              :357 : 9529939547077 us: [pid:31271 tid: 0x145b173466c0] Command (KernelExecution) enqueued: 0x144964749e60 to queue: 0x101fe7b0
:3:rocvirtual.cpp           :883 : 9529939547090 us: [pid:31271 tid: 0x145b173466c0] Arg0:   = val:0x4000000 (size:0x4)
:4:command.cpp              :169 : 9529939547084 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x1449649769a0 complete                                                                             
:4:command.cpp              :169 : 9529939547097 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x14496495dfa0 complete                  
:3:rocvirtual.cpp           :883 : 9529939547093 us: [pid:31271 tid: 0x145b173466c0] Arg1:   = val:0x3 (size:0x1)                                                                  
:3:rocvirtual.cpp           :879 : 9529939547106 us: [pid:31271 tid: 0x145b173466c0] Arg2:   = 0x00 00 60 53 4b 14 00 00 00 00 40 43 4b 14 00 00  (size:0x10)
:3:rocvirtual.cpp           :3351: 9529939547112 us: [pid:31271 tid: 0x145b173466c0] ShaderName : void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scala
r_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor
IteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>)                                                                                                                                      
:3:rocvirtual.cpp           :3549: 9529939547119 us: [pid:31271 tid: 0x145b173466c0] KernargSegmentByteSize = 24 KernargSegmentAlignment = 128
:4:command.cpp              :169 : 9529939547100 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x14496495e240 complete
:4:rocvirtual.cpp           :1083: 9529939547131 us: [pid:31271 tid: 0x145b173466c0] SWq=0x14590ce70000, HWq=0x1457f1700000, id=1, Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1), s
etup=3, grid=[16777216, 1, 1], workgroup=[256, 1, 1], private_seg_size=0, group_seg_size=0, kernel_obj=0x1454f82b71c0, kernarg_address=0x1457f1544280, completion_signal=0x0, correlation_id=0, rptr=39
030, wptr=39805                                                                                                                                                                                        
:4:command.cpp              :169 : 9529939547137 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x144964978020 complete
:4:command.cpp              :169 : 9529939547142 us: [pid:31271 tid: 0x1457f1aaf6c0] Command 0x14496497a990 complete
:3:hip_module.cpp           :813 : 9529939547138 us: [pid:31271 tid: 0x145b173466c0] hipLaunchKernel: Returned hipSuccess : : duration: 86 us 
```
Try to interpret the different kinds of activity.

### 5. Using a profiler to assess GPU activity.

Another way to check for GPU activity is to use a profiler. There is a GPU profiler included in any ROCm instalation: `ROCprofiler`. This profiler is also available inside the containers, so no extra instalations is required. It has a command-line driver called `rocprofv3` and you can see the options one can use with:
```
srun -n1 singularity exec \
    -B .:/workdir \
   /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    rocprofv3 --help
```
Given that Pytorch uses the HIP runtime in its implementation, some of the most relevant options are `--hip-trace`, `--kernel-trace` amd `--memory-copy-trace` to instruct the profiler to collect the HIP runtime, GPU kernel, and copies activity, respectively. Another option that is convinient is `--stats --output-format csv` that generates some statistics on the usage of the GPU and runtime activity and `--output-format pftrace` that generates timelines that can visualized.

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
    trainer.train()
```

Now we can just run the profiler by preceding our original command with `rocprofv3`.

```
srun -n1 singularity exec \
    -B .:/workdir \
    /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'cd /workdir ;  \
             HIP_VISIBLE_DEVICES=0 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --output-format=pftrace -- \
               python -u /workdir/GPT-neo-IMDB-finetuning.py \
                 --model-name gpt-imdb-model \
                 --output-path /workdir/train-output \
                 --logging-path /workdir/train-logging \
                 --num-workers 7'
```

This command would actually fail as some profilling dependencies are not installed in the container. We can install the missing dependencies in a `squashfs` layers and mount it to our container. E.g.:

```
singularity exec \
  -B .:/workdir \
  /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
  bash -c -eux '
    
    mkdir /workdir/deps
    cd /workdir/deps ;
    
    for i in libdw1t64 ; do
      apt-get download $i ;
      dpkg -x $i*.deb . ;
      rm -rf $i*.deb ;
    done
  '

mksquashfs deps deps.sqsh -xattrs-exclude lustre.lov 
```
More on how to extend containers in a later session.

We can now rerun our example with the missing dependency - notice the `--overlay`:
```
srun -n1 singularity exec \
    -B .:/workdir \
    --overlay deps.sqsh:ro \
    /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'cd /workdir ;  \
             HIP_VISIBLE_DEVICES=0 \
             TORCH_HOME=/workdir/torch-cache \
             HF_HOME=/workdir/hf-cache \
             TOKENIZERS_PARALLELISM=false \
             rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --output-format=pftrace -- \
               python -u /workdir/GPT-neo-IMDB-finetuning.py \
                 --model-name gpt-imdb-model \
                 --output-path /workdir/train-output \
                 --logging-path /workdir/train-logging \
                 --num-workers 7'
```

This will generate a file named `nid<node number>/<pid>__results.pftrace`. For example, `nid005024/40587_results.pftrace` will provide the timeline for the execution profilled in node `5024` and process ID `40587`.

### 6. Visualizing a profile trace
To visualize `nid<node number>/<pid>__results.pftrace`, download it to your workstation and load it into the web app `https://ui.perfetto.dev` and will allow you to visualize the GPU execution. Here is a snapshot of the 10 steps executed:

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
    /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif \
    bash -c 'cd /workdir ;  \
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

TIP: JSON is a text format that compresses very well - consider to compress the file prior to copying it to your workstation, it may save you a lot of a time depending on the ammount of activity profiled and your connection speed. 
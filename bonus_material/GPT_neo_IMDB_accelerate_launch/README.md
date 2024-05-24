# HuggingFace accelerate Trainer on LUMI

This is an example of using HuggingFace `accelerate` to launch a LLM training on LUMI making use of the HuggingFace `transforms.Trainer` for setting up the training.

### WARNING: Custom LUMI modules are used in this example

To run these examples, it is assumed that you `module use /appl/local/training/modules/AI-20240529` with installed modules:

- An updated installation of the `cotainr` module that sets `--system=lumi-g` to use the LUMI ROCm base image (/appl/local/`containers/sif-images/lumi-rocm-rocm-5.6.1.sif`)
- The new `singularity-userfilesystems` module that bind mounts user file system paths, i.e. `/project`, `/scratch`, and `/flash`.
- The new `singularity-CPEbits` module that bind mounts the parts of the Cray Programming Environment from the host that are missing in the official LUMI container images due to license restrictions imposed by HPE.

After the AI workshop and the LUMI maintenance break in August/September, these module will hopefully be available in the central LUMI stack.

## 1. Build a container for the example using cotainr

To get started, build a container that includes the conda environment `hf_env.yml` based on the official LUMI ROCm container base image using cotainr.

```bash
module load LUMI cotainr
cotainr build hf_env_container.sif --system=lumi-g --conda-env=hf_env.yml  # or use --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif instead of --system=lumi-g when using cotainr from LUMI
```

More details about building conda/pip environment containers using cotainr may be found in the [cotainr documentation](https://cotainr.readthedocs.io/en/latest/user_guide/conda_env.html) and the [LUMI Docs cotainr documentation](https://docs.lumi-supercomputer.eu/software/containers/singularity/#building-containers-using-the-cotainr-tool).

## 2. Launch the training on multiple GPUs (GCDs) within a single node

To launch the training via `accelerate` on a single node, using all available GCDs, we need to:

- Set the HuggingFace and Torch cache locations, i.e. set the `HF_HOME` and `TORCH_HOME` environment variables.
- Set all "the usual" workarounds needed, e.g. setting the MIOpen DB and cache paths.
- Include all other relevant settings, e.g. bind mounting user files systems, or setting training specific environment variables such as `TOKENIZERS_PARALLELISM`.
- Have an accelerate config file providing the basic configuration, i.e. the parts of the configuration that are fixed and does not relate to the level of scaling of the training or the particular node its running on. Such a config file is provided as `accelerate_hf_trainer_config.yaml`.
- Provide the non-fixed accelerate configuration variables, i.e. number of machines, number of processes, and machine rank, as options to `accelerate launch` when launching the training.

The `train_multi_gpu_hf_trainer.sh` SLURM batch script provides an example of how to launch a training using a single node in the `standard-g` partition based on this approach.

See the [accelerate launch tutorial](https://huggingface.co/docs/accelerate/basic_tutorials/launch), the [accelerate usage guide](https://huggingface.co/docs/accelerate/usage_guides/explore), and the [accelerate cli reference](https://huggingface.co/docs/accelerate/package_reference/cli) for more details.

## 3. Scale to multiple nodes

To scale to multiple nodes, launching the training via `accelerate`, we need to (in addition to the above):

- Apply "the usual" settings and workarounds for a multi node training, i.e. setting `NCCL_SOCKET_IFNAME=hsn`, `NCCL_NET_GDR_LEVEL=PHB` and bind mounting the CPE bits for the `aws-ofi-rccl` plugin.
- Update the non-fixed `accelerate launch` configuration variables, i.e. number of machines, number of processes, machine rank, and main process IP. In particular, we need to pay attention to:
  - The `--num_processes` variable which is the **total** number of processes used in the training across all nodes. This needs to be computed as the number of nodes times the available GCDs/GPUs in each node (8) to have one process for each GCD/GPU.
  - The `--machine_rank` must be set independently on each node, e.g. base on the SLURM node ID - but evaluated on each node.

The `train_multi_node_hf_trainer.sh` SLURM batch script provides an example of how to launch a training using two nodes in the `standard-g` partition based on this approach.

## The magic of HuggingFace

Launching the training using the above approach, the following automagically happens:

- Within the `transformer.Trainer`:
  - [The model is wrapped using PyTorch (Distributed)DataParallel and setup for mixed precision training](https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/trainer.py#L1608).
  - [The distributed training is setup using an `accelerate.Accelerator`](https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/trainer.py#L4322).
- Within the `accelerate.Accelerator`:
  - [The model is transferred to the relevant GPU](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/accelerator.py#L1412)
  - [The dataset batches are transferred to the relevant GPU using a DataloaderDispatcher](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/data_loader.py#L685) which [set up in the `Accelerator`](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/accelerator.py#L1958)
- Within the `accelerate.AcceleratorState`/`accelerate.PartialState` [used in the `accelerate.Accelerator` to define the training environment](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/accelerator.py#L378)
  - [The PyTorch process group is set up](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/state.py#L213) with [backend setup to `nccl`](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/state.py#L730)
  - [The CUDA/ROCm device is set based on the local process index](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/state.py#L784) which is [based on `LOCAL_RANK`](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/state.py#L278) set by [torchrun](https://pytorch.org/docs/stable/elastic/run.html#environment-variables)
- Within `accelerate launch`:
  - [The accelerate configuration variables are converted to the corresponding torchrun options](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/utils/launch.py#L152) 
  - [The training is launched using `torch.distributed.run` (torchrun)](https://github.com/huggingface/accelerate/blob/4ba436eccc1f6437503e66474d8ca86292f4acc1/src/accelerate/commands/launch.py#L733)
- Within `torch.distributed.run`:
  - Sets the [torchrun environment variables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables) somewhere in the torch code base...
  - [Launches the training processes using torch elastic](https://github.com/pytorch/pytorch/blob/6c503f1dbbf9ef1bf99f19f0048c287f419df600/torch/distributed/run.py#L891)

and possibly also a lot of other things as well depending on the launch and `transformer.Trainer` configuration, you use...

## Launching using torchrun instead of accelerate

Launching a `transformers.Trainer` training using `torchrun` instead of `accelerate` only requires calling `torchrun` instead of `accelerate launch` with all `accelerate` configurations converted to the corresponding `torchrun` arguments.

**This is probably the easiest way to launch your `transformers.Trainer` training on LUMI.**

## Launching using Python directly instead of torchrun/accelerate

Launching a `transformers.Trainer` training using `python` directly requires a bit more work, as you need to manually:

- Launch all processes (one for each GPU) using SLURM.
- Set the `RANK` `LOCAL_RANK`, and `WORLD_SIZE`, environment variables for each process such that the `Accelerator` sets up the PyTorch process group correctly.

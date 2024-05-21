import os


def print_slurm_env():
    slurm_env = "\n".join(
        [
            "=" * 80,
            f"SLURM Process: {os.environ.get('SLURM_PROCID', 'N/A')=}",
            "=" * 80,
            f"{os.environ.get('SLURM_NTASKS', 'N/A')=}",
            f"{os.environ.get('SLURM_LOCALID', 'N/A')=}",
            f"{os.environ.get('RANK', 'N/A')=}",
            f"{os.environ.get('LOCAL_RANK', 'N/A')=}",
            f"{os.environ.get('WORLD_SIZE', 'N/A')=}",
            f"{os.environ.get('MASTER_ADDR', 'N/A')=}",
            f"{os.environ.get('MASTER_PORT', 'N/A')=}",
            f"{os.environ.get('ROCR_VISIBLE_DEVICES', 'N/A')=}",
            f"{os.environ.get('SLURM_JOB_GPUS', 'N/A')=}",
            f"{os.sched_getaffinity(0)=}",
            f"{os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', 'N/A')=}",
            "-" * 80 + "\n",
        ]
    )
    print(slurm_env, flush=True)


if __name__ == "__main__":
    print_slurm_env()

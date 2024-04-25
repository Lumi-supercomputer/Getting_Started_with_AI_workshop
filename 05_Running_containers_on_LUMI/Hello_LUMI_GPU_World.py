import os
import sys
import socket

import torch

print(f"Hello LUMI GPU World from {socket.gethostname()}")
print("*" * 80)
print(f" - We are running in the Singularity container {os.environ.get('SINGULARITY_CONTAINER', 'N/A')}")
print(f" - We are running Python version {sys.version} from {sys.executable}")
print(f" - The number of GPUs available to PyTorch is {torch.cuda.device_count()}")
print(f" - Our SLURM job ID is {os.environ.get('SLURM_JOB_ID', 'N/A')}")
print("*" * 80)

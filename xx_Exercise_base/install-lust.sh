#!/bin/bash

module purge

export EBU_USER_PREFIX="/project/project_465001063/lukaspre/EasyBuild"

module load LUMI partition/container
module load EasyBuild-user

mkdir build
cd build
eb ../PyTorch-2.2.0-rocm5.6.1-python-3.10-LUMI-AI-workshop.eb
cd ..
rmdir build

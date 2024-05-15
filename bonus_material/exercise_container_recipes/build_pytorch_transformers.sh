#!/bin/sh

module purge
module use /project/project_465001063/modules
module load LUMI/23
module load cotainr

cotainr build pytorch_transformers.sif --system=lumi-g --conda-env=./pytorch_transformers.yml

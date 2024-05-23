#!/bin/sh

module purge
module use /appl/local/training/modules/AI-20240529/
module load LUMI/23.09
module load cotainr

cotainr build pytorch_transformers.sif --system=lumi-g --conda-env=./pytorch_transformers.yml

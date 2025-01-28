#!/bin/sh

module purge
module load CrayEnv cotainr

cotainr build pytorch_transformers.sif --system lumi-g --conda-env=./pytorch_transformers.yml

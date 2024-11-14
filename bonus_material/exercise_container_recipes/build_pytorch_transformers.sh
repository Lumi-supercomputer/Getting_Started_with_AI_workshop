#!/bin/sh

module purge
#module load cotainr # as of 2024-11-14 the default version of cotainr has a breaking bug
module load cotainr/2023.11.0-20240909

cotainr build pytorch_transformers.sif --system lumi-g --conda-env=./pytorch_transformers.yml

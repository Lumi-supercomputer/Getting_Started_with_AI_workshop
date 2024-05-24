#!/bin/bash
module load LUMI/23.09 systools  # gets us access to proot
singularity build ubuntu_tree.sif ubuntu_tree.def

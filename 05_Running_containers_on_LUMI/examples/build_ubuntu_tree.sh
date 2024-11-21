#!/bin/bash
module load LUMI/24.03 systools  # gets us access to proot
singularity build ubuntu_tree.sif ubuntu_tree.def

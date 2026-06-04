#!/bin/bash
module load LUMI systools  # gets us access to proot
singularity build ubuntu_tree.sif ubuntu_tree.def

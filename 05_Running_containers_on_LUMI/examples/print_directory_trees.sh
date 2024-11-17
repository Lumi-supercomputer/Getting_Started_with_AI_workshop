# Container namespace
echo "Ubuntu container directory tree"
singularity exec ubuntu_tree.sif tree -L 1 /

# LUMI namespace
echo "LUMI directory tree"
module load LUMI/23.09 systools
tree -L 1 /

# Container namespace with /project bind-mounted
echo "Ubuntu container directory tree with /project bind-mounted"
singularity exec --bind /project/project_465001363 ubuntu_tree.sif tree -L 1 /

# 09 Hyper-parameter tuning using Ray

These slides are loosely based on Kaare Mikkelsen's slides (https://www.au.dk/en/mikkelsen.kaare@ece.au.dk) <span style="color:red">*Ask for link to slides*</span>.

## Requirements

First we need to set up a container with ray installed
```bash
module load LUMI/23.03 cotainr
cotainr build minimal_pytorch.sif --base-image=/appl/local/containers/sif-images/lumi-rocm-rocm-5.6.1.sif --conda-env=minimal_ray.yml
```

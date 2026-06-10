# Reference solutions to the hands-on exercises for 06 Building containers from conda/pip environments

## Exercise 1

> 1. Using the example [python312.yml](examples/python312.yml) conda environment, use cotainr to build a container:
>     - Interactively on a login node
>     - Non-interactively on a compute node
> 2. Compare the output of running `python3 -c "import sys; print(sys.executable); print(sys.version)"` on a login node:
>     - In the container you built
>     - Directly on LUMI

To build a container using cotainr on LUMI, we must remember to:

1. Unload the `lumi-aif-singularity-bindings` module. If the module is loaded you will encounter the following error: `FATAL:   container creation failed: mount /var/spool/slurmd->/var/spool/slurmd error: while mounting /var/spool/slurmd: destination /var/spool/slurmd doesn't exist in container`
2. Load the cotainr module on LUMI `module load CrayEnv cotainr`.
3. Determine a suitable base image. For this exercise, we use the `lumi-multitorch-torch-u24r70f21m50t210-20260415_130625.sif` container found in `/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/`.
4. Run cotainr using `srun`, redirect stdout/stderr, and accept all licenses up-front when building non-interactively on a compute node

Since the `python312.yml` environment only contains Python 3.12, we don't need ROCm or other special system libraries. 
Thus, using `--system=lumi-c` instead of `--base-image=...` with cotainr would be sufficient for getting a fairly minimal base image.
However, for sake of consistency we will use the ROCm base image. (Feel free to experiment with the `--system=lumi-c` or `--system=lumi-g` options. However, note that the base images referenced by both `lumi-c` and `lumi-g` are a bit older and you may run into ROCm compatability issues.)

On a login node, we may build the container interactively by:

```bash
$ module load CrayEnv cotainr
$ cotainr build python312.sif --base-image=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-mpich-u24r70f21m50t210-20260415_130625.sif --conda-env=examples/python312.yml
```

On a LUMI-C compute node, we may build the container non-interactively by:

```bash
$ module purge
$ module load CrayEnv cotainr
$ srun --output=cotainr.out --error=cotainr.err --account=project_465002757 --time=00:15:00 --mem=60G --cpus-per-task=32 --partition=debug cotainr build python312.sif --base-image=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-mpich-u24r70f21m50t210-20260415_130625.sif --conda-env=examples/python312.yml --accept-licenses
```

> [!WARNING]
> Cotainr will ask for permission to overwrite the `python312.sif` container if it already exists. Since cotainr currently does not provide a way to non-interactively accept this, when building non-interactively, it will get stuck until it is terminated by SLURM due to the time limit, if the `python312.sif` container already exists.

> [!TIP]
> As an alternative to directly calling `srun`, you may consider creating a SLURM batch script to setup your cotainr build on a compute node.

Now, if we run the `python3 -c "import sys; print(sys.executable); print(sys.version)"` command to show which version of Python we are using, when running in the container, we get:

```bash
$ singularity exec python312.sif python3 -c "import sys; print(sys.executable); print(sys.version)"
/opt/cotainr/conda/envs/conda_container_env/bin/python3
3.12.13 | packaged by conda-forge | (main, Mar  5 2026, 16:50:00) [GCC 14.3.0]
```

whereas directly on LUMI we get:

```bash
$ python3 -c "import sys; print(sys.executable); usr/bin/python3
3.6.15 (default, Sep 23 2021, 15:41:43) [GCC]
```

which shows that within the container we directly have access to the Python 3.12 we installed as part of our conda environment instead of the Python 3.6 provided by the OS. Note that if you run `python3 -c "import sys; print(sys.executable); print(sys.version)"` after having run `module load cotainr`, you will get

```bash
$ python3 -c "import sys; print(sys.executable); /opt/cray/pe/python/3.11.7/bin/python3
3.11.7 (main, Jun 17 2024, 15:36:19) [GCC 12.3.0]
```

since the cotainr module loads the cray-python module to get a Python >= 3.8 which is needed for running cotainr.

## Exercise 2

> 1. Using cotainr, update the container you built using the `python312.yml` conda environment to contain a few extra packages of your choice, e.g. pandas and scikit-learn.
> 2. Open an interactive Python interpreter in the container and import your newly added packages.

To update our container with extra packages, we must remember to:

1. Update the conda environment yaml file and rebuild the container - by design cotainr does not offer a way to change/update an existing container in order to maximize the reproducibility of the software environment in the container and [minimize the risk of ending up with a broken conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
2. Pin versions of the packages we add when updating the conda environment yaml file to maximize reproducibility.

Assuming we would like to add the `pandas`, `scikit-learn`, and `env-var` Python packages to the container, we may create an updated `python312_extra.yml` containing:

```yaml
name: python312_extra
channels:
  - conda-forge
dependencies:
  - pip=24.0
  - python=3.12.3
  - pandas=3.0.3
  - scikit-learn=1.9.0
  - pip:
    - env-var==1.0.1
```

where we have added `pandas`and `scikit-learn` as Conda packages and `env-var` as a pip package, since no conda package exists for it (at least not on conda-forge).

Now we can build the updated container:

```bash
$ module load CrayEnv cotainr
$ cotainr build python312_extra.sif --base-image=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-mpich-u24r70f21m50t210-20260415_130625.sif --conda-env=python312_extra.yml0
```

and open an interactive shell and an interactive Python interpreter in it, and import our added packages:

```bash
$ singularity shell python312_extra.sif
Singularity> python3
Python 3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:38:13) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas, sklearn, env_var
```

> [!NOTE]
> We don't need to activate the conda environment in the container as this is done automatically.

> [!NOTE]
> Even though we pin the versions of the added packages, their dependencies are not pinned and may change if building the container again at a later point in time. To be able to build a new container with the exact same set of all packages (including dependencies), you need to use the output of `conda env export` (in the container) as the conda environment file provided to cotainr (or specify all dependencies manually). The output of `conda env export` in the container looks something like:

```bash
Singularity> conda env export
name: conda_container_env
channels:
  - conda-forge
dependencies:
  - _openmp_mutex=4.5=20_gnu
  - bzip2=1.0.8=hda65f42_9
  - ca-certificates=2026.5.20=hbd8a1cb_0
  - joblib=1.5.3=pyhd8ed1ab_0
  - ld_impl_linux-64=2.45.1=default_hbd61a6d_102
  - libblas=3.11.0=8_h4a7cf45_openblas
  - libcblas=3.11.0=8_h0358290_openblas
  - libexpat=2.8.1=hecca717_0
  - libffi=3.5.2=h3435931_0
  - libgcc=15.2.0=he0feb66_19
  - libgcc-ng=15.2.0=h69a702a_19
  - libgfortran=15.2.0=h69a702a_19
  - libgfortran5=15.2.0=h68bc16d_19
  - libgomp=15.2.0=he0feb66_19
  - liblapack=3.11.0=8_h47877c9_openblas
  - liblzma=5.8.3=hb03c661_0
  - liblzma-devel=5.8.3=hb03c661_0
  - libnsl=2.0.1=hb9d3cd8_1
  - libopenblas=0.3.33=pthreads_h94d23a6_0
  - libsqlite=3.53.2=h0c1763c_0
  - libstdcxx=15.2.0=h934c35e_19
  - libuuid=2.42.1=h5347b49_0
  - libxcrypt=4.4.36=hd590300_1
  - libzlib=1.3.2=h25fd6f3_2
  - narwhals=2.22.1=pyhcf101f3_0
  - ncurses=6.6=hdb14827_0
  - numpy=2.4.6=py312h33ff503_0
  - openssl=3.6.2=h35e630c_0
  - packaging=26.2=pyhc364b38_0
  - pandas=3.0.3=py312h8ecdadd_0
  - pip=24.0=pyhd8ed1ab_0
  - python=3.12.3=hab00c5b_0_cpython
  - python-dateutil=2.9.0.post0=pyhe01879c_2
  - python_abi=3.12=8_cp312
  - readline=8.3=h853b02a_0
  - scikit-learn=1.9.0=np2py312h3226591_0
  - scipy=1.17.1=py312h54fa4ab_1
  - setuptools=82.0.1=pyh332efcf_0
  - six=1.17.0=pyhe01879c_1
  - threadpoolctl=3.6.0=pyhecae5ae_0
  - tk=8.6.13=noxft_h366c992_103
  - wheel=0.47.0=pyhd8ed1ab_0
  - xz=5.8.3=ha02ee65_0
  - xz-gpl-tools=5.8.3=ha02ee65_0
  - xz-tools=5.8.3=hb03c661_0
  - zstd=1.5.7=hb78ec9c_6
  - pip:
      - arrow==1.4.0
      - decorator==5.3.1
      - env-var==1.0.1
      - isoduration==20.11.0
      - rfc3339-validator==0.1.4
      - rfc3986-validator==0.1.1
      - tzdata==2026.2
      - validators==0.18.2
prefix: /opt/cotainr/conda/envs/conda_container_env
```

## Exercise 3

> 1. Create a conda yaml file based on the `python312.yml` and add ROCm 7.0 versions of torch, torchvision, torchaudio and triton-rocm. 
> 2. Build a new container image using cotainr on LUMI-C.
> 3. Confirm that pytorch has access to the AMD GPUs on LUMI-G.


1. You can look for torch wheels [here](https://download.pytorch.org/whl/). The correct link for ROCm 7.0, is: https://download.pytorch.org/whl/rocm7.0/. We need to add this link under the pip section in the conda yaml as an `extra-index-url`. After that we can define the versions of our libraries in the pip section. 


```yaml
name: pytorch
channels:
  - conda-forge
dependencies:
  - pip=24.0
  - python=3.12
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/rocm7.0/
    - triton-rocm==3.6.0
    - torch==2.10.0+rocm7.0
    - torchaudio==2.10.0+rocm7.0
    - torchvision==0.25.0+rocm7.0
```

2. Now we can build a container in the usual way:

```bash
$ module load CrayEnv cotainr
$ cotainr build pytorch.sif --base-image=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-mpich-u24r70f21m50t210-20260415_130625.sif --conda-env=minimal_pytorch.yml
```

You can also find a slightly more extensive `pytorch.yml` in the examples folder. This yaml pins the versions of some of the dependencies that torch installs. 

3. To use the container we first load the `lumi-aif-singularity-bindings` module and then use `srun` with `singularity run`.

```bash
$ module load Local-LAIF lumi-aif-singularity-bindings
$ srun --account=project_465002757 --time=00:15:00 --mem=10G -n 1 --cpus-per-task=1 --partition=dev-g --gpus-per-task=1 singularity run pytorch.sif python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
srun: job 19127979 queued and waiting for resources
srun: job 19127979 has been allocated resources
True
1
```


> [!NOTE]
> You can install packages with conda from Git(Hub) repos. When you install directly from (private) Git(Hub) repos, you may need to install extra Conda packages needed by pip to connect to the repo, e.g. git and openssh. Alternatively, you can also install directly from a zip archive of the repo, e.g. specifying https://github.com/cocodataset/panopticapi/archive/master.zip instead of git+https://github.com/cocodataset/panopticapi.git. See the [cotainr conda environment documentation](https://cotainr.readthedocs.io/en/latest/user_guide/conda_env.html#pip-packages-from-private-repositories) for a more elaborate example.

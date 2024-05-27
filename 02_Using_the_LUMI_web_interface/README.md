# 02 The LUMI web-interface

## Hands-on exercises

1. Get started with the LUMI web interface and set up your own copy of the exercises.

    In this exercise you will gain first experience with using the LUMI web interface to navigate files and directories on the LUMI supercomputer. You will also set up your own copy of the exercise repository on the system, so that you can work on them without interfering with the other course participants.

   1. Log in to the LUMI web interface: https://www.lumi.csc.fi
   2. Create your own scratch subdirectory in `/scratch/project_465001063/`. Use your username for the directory name. You can either
        - Use the built-in file explorer, or
        - Use a login node shell.
   3. Clone the [exercise repository](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop) to your scratch folder `/scratch/project_465001063/<username>`. You can either
        - use a login node shell, or
        - start a Jupyter lab job and use the Jupyter lab UI for cloning Git repositories (check [this blog post](https://blog.reviewnb.com/jupyterlab-git-extension/) for a visual guide for how it works). See point 2.1 below for instructions on how to start the job from the LUMI web interface.
   4. Get familiar with the exercise repository layout.

2. Start an interactive Jupyter lab job and run inference with GPT-neo.

    In this exercise you will learn how to reserve resources for and start an interactive job to run a Jupyter notebook via the LUMI web interface. The notebook itself introduces you to our running example of finetuning a language model using PyTorch and the training libraries provided by Huggingface. In this exercise you will not do any training, but familiarise yourself a bit with the software and the base model.

    1. Start an interactive Jupyter session: Open the Jupyter app (! not "Jupyter for Courses" !) in the LUMI webinterface and set the following settings before pressing `Launch`
        - Project: `project_465001063`
        - Reservation: Use the course reservation `AI_workshop` (there should only be one option)
        - Partition: `small-g`
        - Number of CPU cores: `7`
        - Memory (GB): `16`
        - Time: `0:30:00`
        - Working directory: `/scratch/project_465001063/`
        - Python: `pytorch (Via CSC stack, limited support available)`
        - Virtual environment path: leave empty
    2. Wait for the session to start, then press `Connect to Jupyter`

        > **Note**
        >
        > Jupyter will open in a new tab. Note that your interactive job will not stop if you close the tab, so you can always reconnect to Jupyter via the `My Interactive Session` page of the LUMI web interface. You also have to explicitly `Cancel` the running job from there in order to stop Jupyter when you are done - otherwise your job will continue to consume the allocated resources until the time limit you gave is reached.

    3. Open and run the `<username>/Getting_Started_with_AI_workshop/02_Using_the_LUMI_web_interface/GPT-neo-IMDB-introduction.ipynb` notebook, which introduces our ongoing example for the remaining exercises. Familiarise yourself with the code. You can try to
        - explore the effect of adjusting the parameters for the `tokenizer` and `model.generate` calls
        - try different input prompts
        - explore the contents of the training data set

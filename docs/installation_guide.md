# How to install Cremage

## Linux
Currently, only systems with NVIDIA GPU and CUDA are supported.

### Overview
1. Set up conda environment
1. Install Pytorch
1. Install xformers
1. Install GTK
1. Install other python packages

### Steps

1. Set up conda environment

    Run the following. Note that if your system does not have conda, you need to install it first.

    ```
    conda create -n cremage python=3.10
    ```

2. Install Pytorch
   
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    For more information, refer to the installation section at https://pytorch.org/.

3. Install xformers

    ```
    conda install xformers -c xformers
    ```

4. Install GTK

    Cremage uses PyGObject/GTK3 for UI. Follow the installation instructions at https://gnome.pages.gitlab.gnome.org/pygobject/getting_started.html to set up on your platform.

5. Install other python packages
    ```
    pip install -r requirements.txt
    ```

## Mac (for silicon Mac only)
### Overview
1. Set up conda environment
1. Install Pytorch
1. Install GTK
1. Install other python packages

### Steps
1. Set up conda environment

   Run the following. Note that if your system does not have conda, you need to install it first.

    ```
    conda create -n cremage python=3.10
    ```

2. Install PyTorch

    Go to the installation section at https://pytorch.org/ and follow their instruction to install PyTorch using Conda.

3. Install GTK

    Cremage uses PyGObject/GTK for UI.
    ```
    brew install pygobject3 gtk+3
    ```

4. Install other python packages
    ```
    pip install -r requirements_mac.txt
    ```

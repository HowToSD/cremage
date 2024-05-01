# How to install Cremage

## Linux
Currently, only systems with NVIDIA GPU and CUDA are supported.

### Overview
1. Copy Cremage files from github
1. Set up conda environment
1. Install Pytorch
1. Install xformers
1. Install GTK
1. Install other python packages
1. Copy model files

### Steps
1. Copy Cremage files from github

   Open the terminal and go to a directory that you want to install Cremage.
   Type:
   ```
   git clone https://github.com/HowToSD/cremage.git
   ```

2. Set up conda environment

    Run the following. Note that if your system does not have conda, you need to install it first.

    ```
    conda create -n cremage python=3.10
    conda activate cremage
    ```

3. Install Pytorch
   
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    For more information, refer to the installation section at https://pytorch.org/.

4. Install xformers

    ```
    conda install xformers -c xformers
    ```

5. Install GTK

    Cremage uses PyGObject/GTK3 for UI. Follow the installation instructions at https://gnome.pages.gitlab.gnome.org/pygobject/getting_started.html to set up on your platform.

    For example, for Ubuntu, as described in the above doc, I used below commands to install GTK on my machine:

    ```
    sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-4.0

    pip3 install pycairo
    ```

6. Install other python packages
    ```
    pip install -r requirements.txt
    ```

7. Copy model files
   
   See the section Setting up model files.

## Mac (for silicon Mac only)
### Overview
1. Copy Cremage files from github
1. Set up conda environment
1. Install Pytorch
1. Install GTK
1. Install other python packages
1. Copy model files

### Steps
1. Copy Cremage files from github

   Open the terminal and go to a directory that you want to install Cremage.
   Type:
   ```
   git clone https://github.com/HowToSD/cremage.git
   ```

2. Set up conda environment

   Run the following. Note that if your system does not have conda, you need to install it first.

    ```
    conda create -n cremage python=3.10
    conda activate cremage
    ```

3. Install PyTorch

    Go to the installation section at https://pytorch.org/ and follow their instruction to install PyTorch using Conda.

4. Install GTK

    Cremage uses PyGObject/GTK for UI.
    ```
    brew install pygobject3 gtk+3
    ```

5. Install other python packages
    ```
    pip install -r requirements_mac.txt
    ```

6. Copy model files
   
   See the section Setting up model files.

## Setting up model files
All model files need to be placed under the models directory.
You can also configure the settings to point to the existing directory. Refer to the section "Using the existing model directories".

Here is the structure of the model directory:

| Directory name | Type of model files |
|---|---|
| models/ldm | SD 1.5 base model files |
| models/loras | SD 1.5 LoRA files |
| models/control_net | SD 1.5 ControlNet and IP-Adapter-FaceID files |
| models/embeddings | SD 1.5 textual inversion embedding files |

If you already have models on your system, you can just copy those files to directories above, or change configuration to point to those directories.
If you have never used Stable Diffusion or do not have models on your system,
you can run model_download.sh at the root of Cremage installation directory to download the models.

## Using the existing model directories
If you already have models set up on your system, you can point the settings to those directories. To do so, in Cremage UI, select File | Preferences, and specify directories on your system that correspond to each of the model types.

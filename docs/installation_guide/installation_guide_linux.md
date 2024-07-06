# How to install Cremage for Linux

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

    If this command fails, you may need to try:

    ```
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
    ```
    For more information, refer to https://github.com/facebookresearch/xformers.

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

   See [Setting up model files](setting_up_model_files.md).

8. Start Cremage
   Type:
   ```
   python cremage_main.py
   ```
   This should start Cremage.
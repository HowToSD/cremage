# How to install Cremage for Silicon Mac

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
   
    See [Setting up model files](setting_up_model_files.md).

7. Start Cremage
   Type:
   ```
   python cremage_main.py
   ```
   This should start Cremage.
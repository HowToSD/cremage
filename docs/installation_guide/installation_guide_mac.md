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

8. Change Mac's screen resolution

   Currently Cremage's tools are designed to work with a high resolution display.
   Therefore, if you are using a laptop that has a lower default resolution (e.g. 1440 x 900), some of the tools will not work.  It is recommended to switch to a higher resolution (e.g. 2048 x 1280) using the Displays setting screen in System Settings on your Mac.

Now have a look at [Getting Started](../getting_started.md "View Getting Started") to finish set up in Cremage and generate your first image.

### Troubleshooting
If Cremage does not start or Cremage throws an error, check the following:

#### Model preference is not set
Cremage throws an error if you did not specify the model to use before generating an image. Check Getting Started above for the steps to do so.

#### Wrong PyTorch version
You may have installed a wrong PyTorch version. To check, type the following in terminal:
```
conda activate cremage
python
```
This starts Python. Then in Python, type:
```
import torch
torch.__version__
```

This should show 2.4.0 or later.
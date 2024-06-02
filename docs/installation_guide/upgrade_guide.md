# How to upgrade Cremage

## Update codebase
On your Cremage installation directory, type:
```
git pull --rebase
```

## Install new dependencies
If you have been using Cremage v1.0 for SD 1.5, you need to do the following:

All users:
```
conda activate cremage
pip install ftfy==6.2.0
```

In addition, Window users need to install xformers.
Type:
```
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
```
For more information, refer to https://github.com/facebookresearch/xformers

## Update preferences

Rename config.yaml to config.yaml.sav
```
mv config.yaml config.yaml.sav
```

Start Cremage
```
conda activate cremage
python cremage_main.py
```

Once UI is up, exit without doing anything. This will create a fresh config.yaml file with default settings.

Now you can copy your preferences from config.yaml.sav to config.yaml using a text editor. You can also set your preferences within Cremage UI as well.
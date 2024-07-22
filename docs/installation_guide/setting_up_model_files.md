# Setting up model files
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
you can run model_download.sh at the root of Cremage installation directory to download the models.  If you are using Windows, run model_download.ps1 instead.

## Using the existing model directories
If you already have models set up on your system, you can point the settings to those directories. To do so, in Cremage UI, select File | Preferences, and specify directories on your system that correspond to each of the model types.

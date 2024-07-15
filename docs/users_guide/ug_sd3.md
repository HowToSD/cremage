# How to Use Stable Diffusion 3 with Cremage

Please follow the instructions in the installation or upgrade guide if you haven't done so already. The following steps assume that Cremage is already installed and the UI is accessible.

If you are updating your installation with the new version of the Cremage code, you also need to update the diffusers package version by typing the following:
```
conda activate cremage
pip install diffusers==0.29.2
```

## Setting Up

1. Create an access token at Hugging Face.
   You need to an access token at Hugging Face.
   Create one at https://huggingface.co/settings/tokens

2. Copy model files
   Make sure that you have "git lfs" on your system.
   You can type:
   ```
   git lfs version
   ```
   to see if it's installed. If not install it first.

   Now in a directory where you want to save weights, type:
   ```
   git clone https://<your Hugging Face user ID>:<Hugging Face Access token>@huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
   ```
   Replace <your Hugging Face user ID> and <Hugging Face Access token> with your user ID and your access token.
   Once the files are downloaded, make a note of this diretory.
   
   For example if you download the weights to /home/john_doe/sd3/stable-diffusion-3-medium-diffusers

   Then if you type:
   ```
   ls -1 /home/john_doe/sd3/stable-diffusion-3-medium-diffusers
   ```

   Expected output is:
   ```
   LICENSE
   mmdit.png
   model_index.json
   README.md
   scheduler
   sd3demo.jpg
   text_encoder
   text_encoder_2
   text_encoder_3
   tokenizer
   tokenizer_2
   tokenizer_3
   transformer
   vae
   ```

3. Update config.yaml
   Edit config.yaml at the root of the Cremage installation directory add the following lines:
   ```
   sd3_ldm_model_path: <Path to SD3 weights that you downloaded>
   sd3_ldm_model: None
   ```

   For example:
   ```
   sd3_ldm_model_path: <Path to SD3 weights that you downloaded>/home/john_doe/sd3/stable-diffusion-3-medium-diffusers
   ```

## Generating an image using SD 3
  On Basic tab, select SD 3 in the Generator model type combobox.

## Limitations
* Only text to image flow is supported.  Image to image and Inpainting are not supported.
  However, you can still use Spot Inpainting tool or SD 1.5 Inpainting tool.
  Spot Inpainting tool is accessible from the Tools tab. SD 1.5 Inpainting tool is available when you switch the generator model type to SD 1.5.
* Custom weights are not supported.
* LoRA is not supported.
* FaceID is not supported during generation. However, you can use Face Fix tool to apply FaceID to the generated image. Face Fix tool uses SD 1.5 underneath.
* Auto face fix is supported, but the custom face image that you specify is not used during auto face fix. If you want to use the custom face image, use Face Fix tool after generation.
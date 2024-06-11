# How to Create a Video Using SVD

Please follow the instructions in the installation or upgrade guide if you haven't done so already. The following steps assume that Cremage is already installed and the UI is accessible.

## Setting Up

1. **For users who have already installed the previous version of Cremage:**
   - Edit `config.yaml` and add the following at the bottom of the file:
     ```yaml
     svd_model_path: [Your Cremage installation path]/models/svd
     ```
     Replace `[Your Cremage installation path]` with the actual path on your system.

2. **For users who have already installed the previous version of Cremage:**
   - Activate the Cremage environment and install the necessary package:
     ```bash
     conda activate cremage
     pip install rembg==2.0.56
     ```

3. Access [Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main) to download the SVD 1.1 model. You need to fill out a form as required by Hugging Face.

4. Download `svd_xt_1_1.safetensors` after you fill out the form.

5. Place the model in the `models/svd` directory under the Cremage installation path.

6. Start Cremage.

7. Generate an image or select an image from the image list box on the left.

8. Click the **Tools** tab.

9. Click **Video Generator**.

10. This page shows two images. The one on the left is the thumbnail of the original image. The one on the right is the image that is cropped to match the required SVD resolution (1024x576). If you want to adjust where to crop, click **Adjust Crop** to bring up the cropper tool and select the desired region.

11. Click **Generate**.

12. Check the console output for errors.

13. After the video is generated, it should auto-play in the same window. The generated video is saved under `.cremage/outputs` in your home directory, which you can access by selecting **File | View output directory** in the main Cremage UI.

## Limitations

- This feature is only tested on Ubuntu with NVIDIA 4090 (24GB VRAM). It's unlikely to work with a machine that has less than 24GB VRAM. However, if you want to use this feature on a machine with less VRAM or on a Mac, please create a GitHub issue. I will review and prioritize the ticket.
- The quality of the video may not be optimal if the image contains humans.

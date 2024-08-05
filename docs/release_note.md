# Release Note
August 4, 2024 (v3.2.7)
Face Fix now supports using an SDXL model. This will make it possible to use a higher quality face for improving faces in an image. To use this feature, go to the Tools tab, select "Face fix" to display the Face fix tool. On the tool window, select "SDXL" and the model that you want to use.
Please note:
1. IPAdapter FaceID is not supported for SDXL in Face fix tool yet, so the face image you specified on the Face tab is not reflected.
1. Auto face fix does not support SDXL models yet, so it uses an SD 1.5 model as before.
1. It takes longer to repair a face using SDXL than SD 1.5, so monitor console to check progress.

Improved face detection rate with OpenCV when the input image is large by feeding a scaled-down image into the detector when image size exceeds the certain threshold.

July 31, 2024 (v3.2.6)
PixArt-Sigma txt2img is now working on a silicon Mac. It was tested on MacBook Air (M1, 2020) with 16GB RAM. To use, select Pixart Sigma in Generator model type on Basic tab. Models are automatically downloaded, so no further set up is needed. If you have previously installed Cremage, please make sure that you install sentencepiece as specified in requirements_mac.txt.

July 30, 2024 (v3.2.4):
SDXL Inpainting is now supported. Weights are automatically downloaded from Hugging Face so no user action is required. If you have disabled model downloading network connection in Preferences screen, make sure that you renable this when you use this feature for the first time. To use this, select Inpainting radio button at the bottom of the screen. For more information, refer to [Inpainting](users_guide/ug_inpainting.md "Inpainting")

(v3.2.3):
Added support of copying the file to the favorites directory on Image List on the left of the main image. To copy, press "f" after selecting the image. The image is copied to .cremage/favorites under your home directory.  You can also view this directory by selecting File | View favorites directory menu item from the top menu bar.
In addition, prior to version 3.2.3, marking the current image and going to the marked image required pressing Ctrl+Shift with m or g, but pressing Ctrl+Shift is no longer quired starting in 3.2.3.

(v3.2.2):
Added support for copying positive prompt and negative prompt from generation information to positive prompt and negative prompt fields. To use this feature, just press the "C" icon by one of the prompt fields.
Note that if you just want to use previous generation information including prompts, select "Use generation infor" checkbox and press Generate.  If you want, you can also edit the parameters inside the Generation information textbox.

(v3.2.1):
Fixed the random number seed issue for image generation.

July 26, 2024 (v3.2.0):
Added prompt override, denoising strength, and face detection method fields for the Auto Face Fix feature. These options become available when you enable Auto Face Fix in the Basic tab.
Auto Face Fix applies facial enhancements after generating an image. It detects faces in the image and uses image-to-image processing to improve their appearance. Previously, these options were only available through the Face Fix tool, requiring manual invocation. Now, you can configure these parameters during image generation.
For example, if you're using a Pony model in SDXL and want to make the face look more realistic, you might use a very long prompt with pony-specific words. Since Face Fix uses SD1.5, it’s beneficial to use a shorter prompt with a low denoising value for the face only. This can significantly enhance the overall look.

This is a breaking change and requires your action if you have already installed Cremage. Please add the following lines in config.yaml:
```
auto_face_fix_strength: 0.2
auto_face_fix_prompt: ''
auto_face_fix_face_detection_method: OpenCV
```

July 25, 2024:
For newly added models (e.g., PixArt-Sigma), code has been added to use locally cached model files without establishing a network connection. For some models or features, Cremage may need to download the model initially if you have not used the model or feature before. However, after the initial download, if you turn off "Enable hf internet connection" in the Preferences window, Cremage should not make any Internet connection. On Linux, you can verify this by typing:
```
sudo netstat -apon | grep tcp
```
If you see any connections established after starting Cremage with the connection disabled in the above setting, please file a ticket on the Issue tracker of this repository.

July 20, 2024:
Improved speed for SDXL image generation by optimizing model loading logic.

July 19, 2024:
Stable Cascade support has been added.  No manual set up is required to use this model.
Just select Stable Cascade in the Model generator type combobox on the Basic tab to generate an image.

July 18, 2024:
Hunyuan-DiT support and 900M parameter variant of PixArt-Σ (Pixart Sigma) support have been added.
Fine-tuned models for this variant are not supported yet.
Please see [documentation for PixArt-Σ (Pixart Sigma)](docs/users_guide/ug_pixart_sigma.md "PixArt-Σ (Pixart Sigma)")

This is a breaking change and requires your action if you have already installed Cremage. Please add the following lines in config.yaml:
```
pixart_sigma_model_id: None
```

July 17, 2024:
PixArt-Σ (Pixart Sigma) support has been added. Only text to image flow is supported for now.

This is a breaking change and requires your action if you have already installed Cremage. Please add the following lines in config.yaml:
```
pixart_sigma_ldm_model_path: None
pixart_sigma_ldm_model: None
```

To use this model, select Pixart Sigma from the Generator model type on the Basic tab. Some of the fine-tuned models are also supported. Please see [documentation for PixArt-Σ (Pixart Sigma)](docs/users_guide/ug_pixart_sigma.md "PixArt-Σ (Pixart Sigma)")

July 16, 2024:
Image to image flow and inpainting flow are now supported for Kandinsky 2.2.

July 15, 2024:
Kandinsky 2.2 support has been added.  Only text to image flow is supported for now.
No set up is required to use this model and the model is automatically downloaded. To use this, just select Kandinsky 2.2 from the Generator model type on the Basic tab.

July 14, 2024:
Stable Diffusion 3 (SD3) support has been added.
This is a breaking change and requires your action if you have already installed Cremage. Please add the following lines in config.yaml:
```
sd3_ldm_model_path: <Path to SD3 weights that you downloaded>
sd3_ldm_model: None
```
and update the diffusers package version by typing the following:
```
conda activate cremage
pip install diffusers==0.29.2
```

For more information, please refer to [documentation for Stable Diffusion 3](docs/users_guide/ug_sd3.md "SD3")


July 11, 2024:
Made maximum paint area size configurable for spot inpainting with an inpaint model.
This is a breaking change and requires your action if you have already installed Cremage. Please add the following line in config.yaml:
```
inpaint_max_edge_len: 512
```
Previously, a 512x512 box was selected to wrap your masked region when you spot-inpaint using an inpaint model, but changing this value will allow you to cover the larger area.  However, please be aware that inpainting quality may go down as you increase the size if the inpainting model was not trained with larger image size.

July 11, 2024: Added text prompt safety checker to detect NSFW word(s).
This uses fine-tuned BERT-based classifier model. For details, refer to [documentation for safety checker](docs/users_guide/ug_safety_checker.md "Safety checker").

July 2, 2024: Added LLM interactor to have a discussion about images with an LLM locally. This can be used to analyze any images including ones generated in Cremage as well as generated externally.
Currently, "llava-hf/llava-v1.6-mistral-7b-hf" is used as the LLM model for this feature.
This model is automatically downloaded, so there is no need for you to manually download.
To use, select an image on the image list, and click LLM interactor on the Tools palette.

June 27, 2024: Experimental face colorize feature has been added. This feature is still work in progress and requires some post editing after using the feature. Checkout the [documentation for fixing a face](docs/users_guide/ug_fixing_face.md "Fixing a face") for more information.

June 26, 2024: Experimental face unblur feature has been added. This feature is still work in progress and requires some post editing after using the feature. Checkout the [documentation for fixing a face](docs/users_guide/ug_fixing_face.md "Fixing a face") for more information.

June 11, 2024: Video generation using Stable Video Diffusion (SVD) 1.1 is now supported. For more information, check out [Creating a video using SVD](docs/users_guide/ug_video.md "Video").

At this point, this feature is only available for machines running Ubuntu with 24GB GPU RAM. However, if you want to use it on a host with less RAM, please file a ticket so that I can review and prioritize.

June 6, 2024: Spot inpainter now supports inpaint models in addition to regular SD 1.5 models. Inpaint models have been supported in the main UI, but fixing seams required switching between Spot inpainter and the main UI in the past. Now, you can inpaint using both models within Spot inpainter. This will make seam fixing much easier. Also, for inpainting using an inpaint model, Spot inpainter extracts a 512x512 region surrounding the mask to inpaint instead of processing the entire image, allowing you to touch up a large image.

June 4, 2024: The wildcards feature support has been added to randomly replace a part of the prompt with a predefined set of words. Check out [How to Use Wildcards](docs/users_guide/ug_wildcards.md "Wildcards") for details.  If you have already installed Cremage, this update requires an extra step to manually update your configuration after pulling the latest code from GitHub.
1. Open your text editor.
2. Edit the config.yaml file located in the installation directory of Cremage.
3. Add the following line at the end of the file:
```
wildcards_path: data/wildcards
```

June 3, 2024: Model mixer tool has been added to mix models. This tool works for both SD 1.5 and SDXL.
The tool is available on the Tools tab.

June 2, 2024: Cremage now supports SDXL. Check out [Using SDXL with Cremage](docs/users_guide/ug_sdxl.md "View the SDXL Guide")

May 6, 2024: *BREAKING CHANGE* Made face strength of FaceID configurable. Using a text editor, add the following line at the bottom of config.yaml in the Cremage installation directory.

```
face_strength: 0.7
```

Existing Cremage installation will fail to start without making this change.

May 4, 2024: Experimental segmentation inpainting was added.
[Watch the demo video](docs/videos/segmentation_inpainting_4x_speed.mp4)
(Note the video is 4x speed of the actual).
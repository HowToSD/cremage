![Sample output](docs/images/sample_output_1.jpg "Sample output")
![Sample output](docs/images/sample_output_2.jpg "Sample output")

## Updates
See  [Release note](docs/release_note.md "View Release note").

August 24, 2024 (v4.0.0)
I have undertaken a major refactoring to address a memory leak that occurred when using the auto face fix feature. This bug manifested as an "out of memory" error in system RAM when generating images in SDXL with auto face fix enabled. Due to the extensive code changes, I have released a new version. This update has been tested only on Linux. If you encounter any issues on macOS or Windows, please file a ticket.

# Welcome to Cremage.

Cremage is designed with the following goals in mind:

1. Make it easy to generate images even if you are not familiar with AI.
2. Make it easy for a power user for tweaking new images as well as previously generated images.

For example, if you are starting out with AI image generation, you can just enter a positive prompt and press the Generate button to create images without tweaking any options.

You can also go back to any of the previously generated image and tweak as Cremage provides the same image editing capability irrespective of whether it's newly generated or images generated months ago.

# Supported model types
* FLUX.1-schnell
* Stable Diffusion 1.5 (SD1.5)
* SDXL
* Stable Diffusion 3 (SD3)
* Stable Cascade
* Stable Video Diffusion (SVD)
* Kandinsky 2.2
* PixArt-Σ (PixArt Sigma)
* Hunyuan-DiT

# Major Features
* Text to image (including Stable Diffusion 3)
* Image to image
* Inpainting
* Video generation using SVD including frame interpolation
* Local LLM interaction using Llava-Mistral
* Face fix
* ControlNet
* LoRA
* IP-Adapter-FaceID
* Colorization of face
* Hires fix
* Textual inversion embedding
* Prompt weight
* Wildcards in prompt
* Segmentation inpainting
* Prompt history
* Prompt pre and post expansion (e.g. helpful for PONY models)
* Prompt builder using visual tags
* GFPGAN/RealESRGAN support
* Meta data saving in image file and generating a new image from that
* Model mixing to create a new model from existing models

# High quality face generation
## Bad face fix
Even using a high-quality fine-tuned model, Stable Diffusion can still generate a bad face when the size of the face is relatively small. Cremage offers face fix capability which is inspired by a feature pioneered by Adetailer for Automatic1111. Cremage allows you to improve face during and after image generation, resulting in high quality face images.

<figure>
  <img src="docs/images/bad_faces.png" alt="Example of bad faces">
  <figcaption>Example of bad faces</figcaption>
</figure>

<figure>
  <img src="docs/images/bad_faces_fixed.png" alt="Example of bad faces fixed">
  <figcaption>Example of face fix applied to the image</figcaption>
</figure>

## Consistent face generation
In addition, with fully integrated IP-Adapter-FaceID, you can effortlessly generate the face of a same person consistently across multiple images. This feature is designed to be easy to use and all you have to do is specify a source face image and the desired prompt. Check out a demo video on X by clicking the image below:

[![Watch the demo on X video](docs/images/face_generation_demo.jpg)](https://twitter.com/i/status/1787696937124475046)

## Easy to use as a digital makeup tool
It's easy to use Cremage as a digital makeup tool with integrated ControlNet and img2img.
Below is an example where the original image was processed in Cremage to reduce wrinkles and skin spots:

![Example of digital makeup](docs/images/digital_makeup.jpg "Digital makeup")

## Free Yourself from Tedious Prompt Re-Typing

One of the most time-consuming aspects of image generation is typing prompts. To make it easier, Cremage offers:

* Prompt history
* Prompt expansion
* Generation using prompts from previously generated images

Cremage remembers your prompts each time you generate an image using a different prompt. You can search and select a previous prompt from the history screen by pressing an icon next to the prompt field.

Prompt expansion allows you to define parts of a prompt to be prepended and/or appended to your main prompt. This is particularly helpful when you need to add a long prompt, such as "score_9, score_8_up, score_7_up, rating_safe," before your main prompt.

![Prompt expansion](docs/images/prompt_expansion_ss.jpg "Prompt expansion")

Since different models require different sets of prompts, Cremage also maintains a history for each expansion, allowing you to search and choose as needed. You can easily turn prompt expansion on or off.

Generating images using information from previously generated images helps create similar images. By selecting the "Use generation info" checkbox, the same prompts as the previous image will be used. You can also edit the Generation Information field to tweak the settings.


## Building a Prompt Using Tagged Images

For some models, tags are essential in generating the image that you want, and you have to use the ones that the model was trained with. However, since it is difficult to memorize tags, Cremage offers a fully customizable visual prompt builder.

To use this feature, just put an image file with the tag as the file name under the Cremage data directory. The category of the tag will be the directory name of the image files.

Cremage includes women's clothing tags as a starting point, which you can expand with your own tags.

![Prompt builder](docs/images/prompt_builder_ss.jpg "Prompt builder")


# Installation & Upgrade
Refer to [Installation & Upgrade Guide](docs/installation_guide.md "View the Installation & Upgrade Guide")

# System requirements
* Linux with NVIDIA GPU with at least 8GB of GPU memory
* Microsoft Windows with NVIDIA GPU with at least 8GB of GPU memory
* Silicon Mac

Please note that some features require more GPU memory & system memory, and are only available and/or have been tested on Linux with CUDA.
Mininum recommended system RAM size is 24GB. However, you can still use MacBook Air (16GB RAM model) for a subset of the features. For FLUX.1-schenell, minimum 40GB system RAM is recommended.

# Getting Started
If you have not used any Stable Diffusion software, check out [Getting Started](docs/getting_started.md "View Getting Started") first.

# User's guide
Cremage is designed to make it intuitive to use, but there are still things that are not obvious. I recommend you have a quick look at [User's Guide](docs/users_guide.md "View the User's Guide")

# Usage Restrictions

## Non-commercial Use
**Cremage** is initially released for non-commercial research purposes only; however, this status is subject to change. If you are interested in using it commercially, please contact me at support@candee.ai.

## Not Suitable for Minors or Those Sensitive to Potentially Offensive Content

While **Stable Diffusion** is a powerful and versatile tool for image creation, it has the potential to generate content that may be inappropriate for minors. Additionally, **Stable Diffusion** can occasionally produce images that some users might find offensive.

The **Cremage** software includes a feature designed to filter out sensitive content; however, this filter is not foolproof and may not effectively screen all such content. As a result, **Cremage** is not suitable for use by minors.

By choosing to use **Cremage**, users acknowledge and accept the risks associated with potential exposure to inappropriate or offensive content.

## Lawful and Ethical Purposes Only

Users are reminded that **Cremage** must be used only for lawful and ethical purposes. This includes refraining from using the software to generate images of any real person without the explicit consent of the individuals whose likenesses are to be used. Users assume full responsibility for ensuring their use of the software complies with all applicable laws and ethical standards.

# Reporting bugs
The best way to report a bug is to create an issue on GitHub.

I hope you will enjoy generating artwork!

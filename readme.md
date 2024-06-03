
## Updates
June 2, 2024: Cremage now supports SDXL. Check out [Using SDXL with Cremage](docs/users_guide/ug_sdxl.md "View the SDXL Guide")

---
# Welcome to Cremage.

Cremage is designed with the following goals in mind:

1. Make it easy to generate images even if you are not familiar with AI.
2. Make it easy for a power user for tweaking new images as well as previously generated images.

For example, if you are starting out with AI image generation, you can just enter a positive prompt and press the Generate button to create images without tweaking any options.

You can also go back to any of the previously generated image and tweak as Cremage provides the same image editing capability irrespective of whether it's newly generated or images generated months ago.

# High quality face generation
## Bad face fix
Even using a high-quality fine-tuned model, Stable Diffusion can still generate a bad face when the size of the face is relatively small. Cremage offers face fix capability which is inpired by a feature pioneered by Adetailer for Automatic1111. Cremage allows you to improve face during and after image generation, resulting in high quality face images.

<figure>
  <img src="docs/images/bad_faces.png" alt="Example of bad faces">
  <figcaption>Example of bad faces</figcaption>
</figure>

<figure>
  <img src="docs/images/bad_faces_fixed.png" alt="Example of bad faces fixed">
  <figcaption>Example of face fix applied to the image</figcaption>
</figure>

## Consistent face generation
In addition, with fully integrated IP-Adapter-FaceID, you can effortlessly generate the face of a same person consistently across multiple images. This feature is designed to be easy to use and all you have to do is specifying the a source face image and the desired prompt.  Check out a demo video on X by clicking the image below:

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


# Installation & Upgrade
Refer to [Installation & Upgrade Guide](docs/installation_guide.md "View the Installation & Upgrade Guide")

# System requirements
* Linux with NVIDIA GPU with at least 8GB of GPU memory
* Microsoft Windows with NVIDIA GPU with at least 8GB of GPU memory
* Silicon Mac

# Getting Started
If you have not used any Stable Diffusion software, check out [Getting Started](docs/getting_started.md "View Getting Started") first.

# User's guide
Cremage is designed to make it intuitive to use, but there are still things that are not obvious.  I recommend you have a quick look at [User's Guide](docs/users_guide.md "View the User's Guide")

# Usage Restrictions

## Non-commercial Use
**Cremage** is initially released for non-commercial research purposes only; however, this status is subject to change. If you are interested in using it commercially, please contact me at support@candee.ai.

## Not Suitable for Minors or Those Sensitive to Potentially Offensive Content

While **Stable Diffusion** is a powerful and versatile tool for image creation, it has the potential to generate content that may be inappropriate for minors. Additionally, **Stable Diffusion** can occasionally produce images that some users might find offensive.

The **Cremage** software includes a feature designed to filter out sensitive content; however, this filter is not foolproof and may not effectively screen all such content. As a result, **Cremage** is not suitable for use by minors.

By choosing to use **Cremage**, users acknowledge and accept the risks associated with potential exposure to inappropriate or offensive content.

## Lawful and Ethical Purposes Only

Users are reminded that **Cremage** must be used only for lawful and ethical purposes. This includes refraining from using the software to generate images of any real person without the explicit consent of the individuals whose likenesses are to be used. Users assume full responsibility for ensuring their use of the software complies with all applicable laws and ethical standards.

# Previous Updates
May 4, 2024: Experimental segmentation inpainting was added.
[Watch the demo video](docs/videos/segmentation_inpainting_4x_speed.mp4)
(Note the video is 4x speed of the actual).

May 6, 2024: *BREAKING CHANGE*  Made face strength of FaceID configurable. Using a text editor, add the following line at the bottom of config.yaml in the Cremage installation directory.

```
face_strength: 0.7
```

Existing Cremage installation will fail to start without making this change.

# Reporting bugs
The best way to report a bug is to create an issue on github.

I hope you will enjoy generating artwork!
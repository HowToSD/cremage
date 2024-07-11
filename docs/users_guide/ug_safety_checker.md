# Safety Checker

## Disclaimer

While **Stable Diffusion** is a powerful and versatile tool for image creation, it has the potential to generate content that may be inappropriate for minors. Additionally, **Stable Diffusion** can occasionally produce images that some users might find offensive.

The **Cremage** software includes safety checkers designed to filter out NSFW content; however, the filter is not foolproof and may not effectively screen all such content. As a result, **Cremage** is not suitable for use by minors.

By choosing to use **Cremage**, users acknowledge and accept the risks associated with potential exposure to NSFW content.

## Types of Safety Checkers

Cremage has two safety checkers for the following:
* Text prompt
* Images

The text prompt check is done when you press the Generate button. The prompt is sent to a locally loaded model, and if the model classifies the text as containing NSFW content, Cremage displays an error message and avoids image generation.

However, a Stable Diffusion model can generate NSFW images even without NSFW tags, or the text checker model can make a classification error. Cremage also contains an image check.

The image check is done after an image is generated. If the checker determines that the image contains NSFW material, a black image is returned.

Even with these two checking mechanisms, NSFW material can still fall through these filters and get displayed. Therefore, as a user, you need to be aware of this possibility.

## How to Turn On/Off Safety Checker

Go to the File | Preferences menu and set the Safety Check field.

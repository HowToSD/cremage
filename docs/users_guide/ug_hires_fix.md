# How to use Hires fix

If you are new to Stable Diffusion and have never used Hires fix, check out my tutorial [How to use Hires fix in Automatic1111](https://howtosd.com/how-to-use-hires-fix-in-automatic1111/ "Hires fix tutorial"). Though this tutorial is written for Automatic1111, it covers the concept to give you a good idea about why it's useful.

To use Hires fix in Cremage, go to Basic tab.
Then select Lanczos or Latent from Hires fix upscaler drop down list.
If you select Lanczos, specify a low denoising value (e.g. 0.2 to 0.3) in Denoising Strength. If you use Latent, specify a higher denoising value (e.g. 0.7).
Note that specifying a higher denoising value can make the image deviate from the original image before upscaling, so you might want to try Lanczos first.
Enter a prompt and press Generate to see the image.
Currently, images are upscaled by 2x using this feature.
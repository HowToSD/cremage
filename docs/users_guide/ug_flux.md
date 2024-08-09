# How to use FLUX.1-schnell

## Setting Up
Make sure that you update dependencies as it requires the newer version of diffusers package.
Weights are automatically downloaded.

## UI
1. Go to the Basic tab.
1. Select FLUX.1-schnell on the Generator model type
1. Specify Sampling steps: 4
1. Specify Cfg: 0
1. Enter prompt and press Generate

If your machine has more than 64GB system RAM and 16GB GPU memory, you can set Low mem to false and Keep instance to true. Doing so can drastically reduce the image generation time.
On a host with 64GB RAM & NVIDIA 4090, 1024x1024 image generation time shrinks from 18 sec to 8 sec. Note when this combination is used, initial generation can take more than a minute due to initialization, but subsequent image generation goes much faster.

## Limitations
- Large System RAM required: This feature is designed to run with a smaller amount of GPU memory and it should run on NVIDIA GPU with 8GB memory.
  However, it requires a large amount of system memory (> 35GB), so if you have less than 40GB (with some buffer), it may not run.
- Negative prompt is not supported.
- Batch size greater than 1 is not supported.
- Auto face fix is not supported.
- Mac is not supported.
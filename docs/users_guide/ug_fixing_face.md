# How to fix a face
There are two ways to fix a face in an image.

* During generation
* After generation

## During generation
Check Auto face fix checkbox. This will turn on face fix for each image generation.
One caveat is that you have less control over how fixes are made, especially
if the image contains multiple faces.

## After generation
First, select an image to fix face from the image list on the left.  If you are fixing the image you just generated, then the image is already selected and is displayed in the main image area.
Go to the Tools tab and click Face Fix.
This brings up the Face Fix tool.

If you want to fix an anime face, select Insight Face from the Detection method combobox. If you want to fix a realistic face, you can keep OpenCV.

First, you can click Auto Fix. This detects all the faces and repairs automatically.

If it does not work, try the following:
Click Detect faces to see if one of more faces are detected. If you are using OpenCV for a realistic face, select Insight Face from the Detection method combobox and try again. If faces are selected, then click Fix marked faces.

If faces are not selected, drag your mouse to draw a square on the face to manually select the face. Then click Fix marked faces.

### If you have multiple faces
Auto Fix can handle multiple faces, but there may be cases where you want to apply different prompts to make each face more distinguishable from each other.
To do so, first click Detect faces to detect faces.  Then click one of the faces. Then click Delete mark button. typing a prompt in Positive prompt override, then click Fix marked faces to apply face fix to the remaining selected face.  Once the face fix is applied, repeat the same process for the other face.

## Denoising strength
Default is 0.3, but this may be too strong. Reduce it to 0.1 or 0.2.

## Going back to the very original image
For some reason you may not like the result and you want to restart from the very original image.  If that's the case, close the Face Fix tool and select the original image on the image list box.

## Cleaning up the artifact
Sometimes you may see an artifact around the face after the fix. The best way to to clean up is use Inpainting.
See the section below Fixing an artifact using inpainting.

# Fixing an artifact in an image using Inpainting
To fix an artifact (e.g. seams in the image or some parts of the image not matching the rest after face fix), there are two ways to do so:
* Inpainting in Inpainting mode
* Spot inpainting tool

## Spot inpainting tool
1. Select an image on the image list box.
2. On the right side of the UI, select Tools tab | Spot inpainting.
3. Make sure Use inpainting model is selected. If not, select it.
4. Paint on the seam with some margins on both sides of the seam.  Note that currently, the maximum area that you can apply inpainting with the inpaint model is 512x512 px.
5. In the prompt, put a generic prompt such as "skin", "hair" based on the area where the seam is located.
6. Click Apply inpainting. This should fix the seam. Repeat these steps if you have multiple seams.

## Inpainting in Inpainting mode
Inpainting mode is available only in SD 1.5, so if you are using SDXL, you need to switch to SD 1.5 on the Basic tab first. Then take the following steps:

1. Select an image on the image list box.
2. Select inpainting radio button
3. Click Copy from main image area. This will copy the image to the left box below the main image area.
4. Now you see a gray box to the right of the copied image. The right box says: "Click to open the mask editor". Click this box. This will bring up the mask editor.
5. In mask image editor, paint on the seam with some margins on both sides of the seam.
6. Click Save and close the editor by clicking the x on the top right of the window.
7. You should see the mask reflected in the right box.
8. In the prompt, put a generic prompt such as "skin", "hair" based on the area where the seam is located.
9. Click Generate. This should fix the seam. Repeat these steps if you have multiple seams.

# Face Unblur and Face Colorize

Face Unblur is an experimental features designed to increase the sharpness of slightly blurred faces. Face colorize is an experimental feature to colorize the face found in a black and white photorealistic image. To use these features, simply click "Unblur Face" or "Colorize Face" in the Face Fix window. 
These features are only available for a machine with an NVIDIA GPU, and are still a work-in-progress, two types of post-editing are required:

- Removing artifacts around the face
- Removing artifacts on the forehead

## Removing Artifacts Around the Face

Bothe features work by extracting a rectangle containing the face from the original image and enhancing it using a different machine learning model. This process affects the entire region that was processed, and when the enhanced region is pasted back into the original image, it may create a visible seam. You can apply the same seam-fixing technique as described above to remove the seam.

## Removing Artifacts on the Forehead

Currently, an artifact may appear on the forehead after processing. You can use inpainting steps to remove this artifact.

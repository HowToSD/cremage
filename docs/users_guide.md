# User's guide

This guide focuses on what is not so obvious from the UI.

# Nagivating previously generated images
On the left of the UI, you see a list box of images.  To navigate, first click an image
on the list, then press any key in the table below:

| Key | Function |
|---|---|
| Home or Cmd+up arrow for Mac | Jump to the most recent image |
| End  or Cmd+down arrow for Mac| Jump to the oldest image |
| Up arrow | Scroll to the newer image |
| Down arrow | Scroll to the older image |
| Shift+Ctrl+m | Mark the current image |
| Shift+Ctrl+g | Go to the marked image |
| Del | Erase current image |
| Backspace | Erase current image |

Once you select an image on the list, the selected image appear on the main image area, which you can edit by various tools.

# How to fix face
There are two ways to fix face.

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
First, you can click Auto Fix. If it does not work, try the following:

Click Detect faces to see if one of more faces are detected. If not, select Insight Face from the Detection method combobox and try again.
If faces are selected, then click Fix marked faces.
If faces are not selected, drag your mouse to draw a square on the face to manually select the face. Then click Fix marked faces.

### If you have multiple faces
First click Detect faces to detect faces.  Then click one of the faces. Then click Delete mark button.  Then click Fix marked faces to apply face fix to the remaining selected face.  Once the face fix is applied, repeat the same process for the other face.

### Prompt
If you have multiple faces, typing a prompt in Positive prompt override field that matches the face will help.

## Denoising strength
Default is 0.3, but this may be too strong. Reduce it to 0.1 or 0.2.

## Going back to the very original image
If you want to restart from the very original image, close the Face Fix tool and select the original image on the image list box.

## Cleaning up the artifact
Sometimes you may see an artifact around the face after the fix. The best way to to clean up is use Inpainting.
See the section below Fixing an artifact using inpainting.

# Fixing an artifact in an image using Inpainting
To fix an artifact (e.g. seams in the image or some parts of the image not matching the rest after face fix), take the following steps:

1. Select an image on the image list box.
1. Select inpaint radio button
1. Click Copy from Main
1. Click the mask image view
1. On mask image editor, paint on the seam with some margins on both sides of the seam.
1. In prompt, put a generic prompt such as "wall", "skin", "hair" based on the area where the seam is located.
1. Click Generate. 
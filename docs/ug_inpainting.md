# Inpainting
Inpainting is an effective to fix an undesired object or an artifact (e.g. seams in the image or some parts of the image not matching the rest after face fix).

Follow below steps to apply inpainting to your image:

1. Make sure that an inpainting model is selected in Models tab | Ldm inpaint model combo box.
1. Select an image on the image list box.
1. Select inpainting radio button
1. Click Copy from main image area. This will copy the image to the left box below the main image area.
1. Now you see a gray box to the right of the copied image. The right box says: "Click to open the mask editor". Click this box. This will bring up the mask editor.
1. In mask image editor, paint on the seam with some margins on both sides of the seam.
1. Click Save and close the editor by clicking the x on the top right of the window.
1. You should see the mask reflected in the right box.
1. In prompt, put a generic prompt such as "skin", "hair" based on the area where the seam is located.
1. Click Generate. This should fix the seam. Repeat these steps if you have multiple seams.
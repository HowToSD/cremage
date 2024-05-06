# How to generate a consistent face
Generating the same face or similar faces across multiple images had been a challenge for Stable Diffusion. One way to effectively address this issue is [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID "IP-Adapter-FaceID"). Cremage has IP-Adapter-FaceID-PlusV2 fully integrated and offers easy-to-use consistent face generation feature.

To use, follow below steps.

1. Make sure that you have downloaded ip-adapter-faceid-plusv2_sd15.bin and copied into models/control_net directory.  You can use the model_download.sh script located the root of the installation directory to do this. (For Windows, use model_dowload.ps1).
2. Click Face tab on the right.
3. Click the gray area on the tab to open the file chooser, and select an image file.
   If you want to use an image in the main image area, press Copy from current image.
   For Linux, you can also drag and drop from OS file viewer.
   The image should appear in the box.
4. Enter a prompt and press Generate.

If you want to disable FaceID, you can check "Disable face input" to do so without deleting the face image. If you want to delete the face image, press "X" under the image.
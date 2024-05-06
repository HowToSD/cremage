# How to use ControlNet

First, put your ControlNet 1.0 and/or ControlNet 1.1 model files in model/control_net under the Cremage installation directory and specify this directory in Preferences menu (File | Preferences).
Then go to the ControlNet tab.

In ControlNet type drop down list, select the ControlNet model that you want to use.
The click the gray box to display ControlNet Input Image Creator (annotator).
In the Creator dialog, select the ControlNet type in ControlType drop down list, and click the left gray box to pick the source image.  Click Generate to generate the ControlNet input image. The generated image is automatically reflected on the ControlNet tab of the main UI. Close the creator and click Generate in the main UI.

# Graffiti editor
You can use Graffiti editor to draw edge or lineart-based ControlNet images.

# Limitations
* Multiple ControlNet is not supported yet.
* Only ControlNet 1.0 annotation is supported in ControlNet Input Image creator,
  but most ControlNet 1.1 models should work if you can provide an annotated input image.
* On Mac, some of the ControlNet 1.0 annotators do not work.
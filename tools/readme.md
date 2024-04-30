README for Tools

Some rules
* Each script should work standalone
* UI should have a button that would generate an image. The image will be saved
  in the output directory. The file name is given by the caller.
  The user can optionally save the file using a different name by selecting
  File | Save.
* When the image is generated, it should call the callback function.
  Callback function contains the following params:
  ** the PIL image object
  ** Generation parameter (serialized json)
  Both of these are used to update the main image and its generation parameters.
# How to use textual inversion embedding

1. Put your SD 1.5 embedding files in model/embeddings under the Cremage installation directory. You may need to create this directory if it doesn't exist already.
1. In Cremage, select File | Preferences, specify this directory in Embedding path.
1. If you use SDXL embeddings, then create a new directory model/embeddings_sdxl.and put SDXL embedding files in that directory.  Note that SDXL embedding files are different from SD 1.5 embedding files.
1. In Cremage, select File | Preferences, specify this directory in Sdxl embedding path.
1. In your prompt, enclose the embedding file name in angle brackets followed by the word "embedding:". For example, if the file name is "awesome.pt", here is an example prompt:

```
A photo of a soccer player in the style of <embedding:awesome.pt>
```

You can use embedding in both positive prompt and negative prompt.

As it is difficult to remember the right embedding file name, Cremage offers an easy way to add embedding to your prompt.

Here are the steps:
1. Select the Tools | TI Embedding menu item. It should show all the embeddings for the currently selected model type (SD 1.5 or SDXL. You can change the model type on the Basic tab). Initially, you see a gray icon for each of the embedding. As you generate an image, the icon will be replaced by a thumbnail of the actual image where the embedding is used.
1. Click Add to positive prompt or Add to negative prompt button to add the embedding to your prompt.

Currently these thumbnail files are stored in .cremage/data/embedding_images under **your home directory**. Note that each thumbnail gets updated everytime when new image is generated with the embedding so even if you replace the file manually, your changes will be overwritten.
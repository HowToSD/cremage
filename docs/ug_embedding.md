# How to use textual inversion embedding

First, put your embedding file in model/embeddings under the Cremage installation directory and specify this directory in Preferences menu (File | Preferences).
Then in your prompt, enclose the embeddig file name in angle brackets followed by the word "embedding:". For example, if the file name is "awesome.pt", here is an example prompt:

```
A photo of a soccer player in the style of <embedding:awesome.pt>
```

You can use embedding in both positive prompt and negative prompt.
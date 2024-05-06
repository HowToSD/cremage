# How to use a LoRA

First, put your LoRA model file in model/loras under the Cremage installation directory and specify this directory in Preferences menu (File | Preferences).
Then go to the Models tab. You will see 5 slots for LoRA models. Pick any slot, select the LoRA model that you want to use and specify the desired weight.
Then in your prompt, put a trigger word as specified by the LoRA model creator.
For example, if the trigger word is "cute puppy", here is an example prompt:

```
A photo of a cute puppy.
```
Note that position of the trigger word(s) may matter depending on the way the LoRA model was trained. The creator may have put the trigger words at the beginning like below, so refer to the creator's guide.
```
cute puppy, a realisic image.
```

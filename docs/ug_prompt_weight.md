# How to specify a weight in a prompt
You can specify custom weight for a word or words in your prompt. To do so,
Select a word or words to modify the weight and enclose them in parentheses.
Then before the closing parenthesis, put the weight for the words preceded by a colon.
Here is an example:
```
A photo of a (soccer player:1.2) playing in a field, wearing a blue uniform.
```

You can nest parentheses, and weights will be multiplied properly.
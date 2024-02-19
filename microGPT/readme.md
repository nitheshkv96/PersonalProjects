# Micro Generative Pretrained Transformer(GPT) Model trained on Shakespeare Dataset!!


**Author: Nithesh**

**Date: 6/20/2023**

## Introduction:
This project is aimed at building a generative pretrained model from scratch based on the decoder architecture of the Transformer model and a small attempt to learn the model architecture and math involved in it.
The model is trained on a small Shakespeare dataset with an objective to generate a character level text that closely resemble the Shakespeare way of writing poem.

## Tools Used:
Spyder
Google Colab

## ML Frameworks:
PyTorch with Cuda enabled

## Conclusion:
Model has learned to generate meaningful English words to a great extent. However there is more room for improving the accuracy of the model to generate more meaningful and sensible texts.

## Future Work:
Currently the vocab size is pretty small because of character level tokenization of the entire data. In my upcoming work I will try to assume a subword level tokenization technique creating a largetr vocabulary size and also increase the model size with some hyperparameter tuning to achieve a desired level of model accuracy in genreating texts.

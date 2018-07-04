Lip Reading in the Wild using ResNet and LSTMs in Torch
=======================================================

This repository contains the code I used to train and evaluate (most of) the models described in [Combining Residual Networks with LSTMs for Lipreading](https://arxiv.org/pdf/1703.04105.pdf) by T. Stafylakis and G. Tzimiropoulos

The code is based on facebook's implementation of [ResNets](https://github.com/facebook/fb.resnet.torch)

## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Install [rnn](https://github.com/Element-Research/rnn) (not tested with more recent versions).
- Download the [Lip Reading in the Wild](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) dataset

## Training

The training scripts come with several options, which can be listed with the `--help` flag.

This is the suggested order to train the models:

(i) Start by training a model with temporal convolutional backend (set `-netType 'temp-conv'`). Set `-LR 0.003` and let it for about 30 epochs.
(ii) Throw away the temporal convolutional backend, freeze the parameters of the frontend and the ResNet and train the LSTM backend (set `-netType 'LSTM_init'`). 5 epochs are enough to get a sensible initialization of the LSTM. Set `-LR 0.003`
(iii) Train the whole network end-to-end (set `-netType 'LSTM'`). In this case, set `-LR 0.0005` and about 30 epochs.

The (i) should yield about 25% error rate and (iii) about 17%.

All these steps are performed (semi-)automatically by the code. You should (a) change the `netType` and `LR` parameters and (b) set the `retrain` parameter to the path where the previous model is stored. For (i), set `retrain` to `none`.

I used a single GPU without any of the memory optimization methods of the original ResNet (e.g. shareGradInput, optnet).

## Pretrained models

Please send me an email at themos.stafylakis@nottingham.ac.uk or at themosst@gmail.com.

## Examples of LRW and how to evaluate on them

In fast_evaluation you will find evaluate_examples.lua, together with some files (in torch format) from LRW and its vocabulary (500 words). Run the script and verify that (at least most of) the 5 examples are correctly classified. The .t7 files are also useful in order to check how the input of the ResNet should look like.  

## Number of frames

The number of frames per clip is 29. In the paper we refer to 31 because I used an older version of `ffmpeg` to extract images, that (for some unknown reason) prepends two copies of the first frame.

## Landmark Detection

In my original implementation I used landmark detection, based on which I was estimating the boundaries of the mouth region. However, one can skip this step and crop the frames using a fixed window (see `datasets/BBCnet.lua`) since the faces are already centered. 

## Model Parameters, SoftMax and pooling

In the paper I used the 34-ResNet, although 18-ResNet performs equally well. You can play a bit with other parameters, such as inputDim and hiddenDim, or the activation function.

Moreover, it would be interesting to try batchnorm and/or dropouts in the BiLSTM. 

I use one SoftMax per BiLSTM output, but I have also tried using average pooling combined with a single SoftMax. 
I tried SoftMax on the last frame as well (the latter did not do well with unidirection LSTM, but was OK with BiLSTMs). 
I didn't notice any substantial difference between the three approaches.

## Word Boundaries

Currently, the models do not make use of the word boundaries that are provided with the dataset. However, I will soon upload code that makes use of them. The performance is about 12.7% error rate, compared to 17.0%.

## Deep word embeddings for visual speech recognition

It is largely based on this code, with some differences mainly on the backend and on the use of word boundaries. 





 

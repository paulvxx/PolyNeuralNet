# PolyNeuralNet

## High Level Description

This repository focuses on training Neural Networks with Learnable Polynomial Activations for degrees 2, 3, 4. 
We train image classifiers on the datasets [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) [1] 
and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) [2].

For Shallow Neural Networks, we use the following architectures:
- An MLP with an initial input layer of 784 neurons, and a hidde layer with 512 neurons, and an output layer with 10 neurons [should I cite this?  https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html or this https://github.com/NvsYashwanth/Fashion-MNIST-Image-Classification]. This setup follows the Pytorch Tutorial setup, though we use different optimizations and allow for general non-linear activation functions.

![Model architecture](assets/architecture.png)

## Sources

[1] Han Xiao, Kashif Rasul, and Roland Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747 [cs.LG], 2017.  
[2] Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Technical Report, University of Toronto, 2009.

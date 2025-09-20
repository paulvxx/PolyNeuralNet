# PolyNeuralNet

## High Level Description

This repository focuses on training Neural Networks with Learnable Polynomial Activations for degrees 2, 3, 4. 
We train image classifiers on the datasets [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) [1] 
and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) [2].


## Architectures

For Shallow Neural Networks, we use the following architecture:
- The MLP consists of an input layer of 784 units (corresponding to the flattened 28×28 Fashion-MNIST image), one hidden layer with 512 units, and an output layer with 10 units producing the class logits. The architecture is illustrated in the diagram below.

![Model architecture](assets/ShallowMLP.png)

## Activation Function

The choice of activation function in the MLP architecture is a central focus of this study. In our case, we compare four different activations:

- Quadratic : $f(x) = ax^2 + bx + c$
- Cubic : $f(x) = ax^3 + bx^2 + cx + d$
- Quartic : $f(x) = ax^4 + bx^3 + cx^2 + dx+ e$
- ReLU : $f(x) = \max(0,x)$   [used for benchmark comparisons only]

## Initialization



## Experiment



## Sources

[1] Han Xiao, Kashif Rasul, and Roland Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747 [cs.LG], 2017.  
[2] Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Technical Report, University of Toronto, 2009.

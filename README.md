# NeuralNotwork
This repo is like my playground for neural networks. Here, I take a shot at implementing
some neural network ideas I've come across or cooked up myself. 
As the repo name implies, not all of these ideas turn out as expected, and that's all part of the adventure :)

My goal here is to document my exploration and get a feel for what 
neural networks are capable of, let me know if you had any suggestions.

Each folder contains an implementation of some ideas. Here's a quick overview
of what you can find within each one:

+ <b>basic</b>: The first project in the repo, where I took a shot at implementing a few NN layers, some manual backprop, and Adam optimization.
+ <b>calc</b>:  An experiment where I'm toying with the idea of introducing memory to transformers, demonstrated through a calculator. (WIP)
+ <b>MNIST</b>: Trying out ConvNets on MNIST dataset then trying knowledge distillation on a smaller net to assess its performance against raw data training.
+ <b>CAFA-tree</b>: A shot at solving the CAFA challenge, although it didn't hit the mark, The challenge revolves around associating proteins with their GO annotations, and my approach was predicting the next layer of the tree-like structure of GO annotations in each step using a transformer decoder. The idea was fun but the results were 💀

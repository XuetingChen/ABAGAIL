CS7641: Assignment 2
====================

Code and datasets for Georgia Tech's CS7641 Spring 2016 Randomized Optimization project.

This analysis used ABIGAIL running on Mac OS X, available from this Java repo.

(Forked from the [original](https://github.com/pushkar/ABAGAIL/).)

The Code
--------

To run from a terminal:

1. Go to the root of the repo directory.
2. Run: `ant`
3. Run: `java -cp ABAGAIL.jar opt.test.NurseryTest`
4. Run: `java -cp ABAGAIL.jar opt.test.OptimizationTest`


The Data
--------

The data sets used by the neural network trainer for the Nursery problem can be found under: `data/`

The Results
-----------

After running the above code steps:

* Nursery problem results will be under: `NN_Results/`
* Optimization problem results will be under: `Optimization_Results/`

(Original README follows)


ABAGAIL
=======

The library contains a number of interconnected Java packages that implement machine learning and artificial intelligence algorithms. These are artificial intelligence algorithms implemented for the kind of people that like to implement algorithms themselves.

Usage
------

See [Wiki](https://github.com/pushkar/ABAGAIL/wiki)

Issues
-------

See [Issues page](https://github.com/pushkar/ABAGAIL/issues?state=open).

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_branch`)
3. Commit your changes (`git commit -am "Awesome feature"`)
4. Push to the branch (`git push origin my_branch`)
5. Open a [Pull Request][1]
6. Enjoy a refreshing Diet Coke and wait 

Features
========

### Hidden Markov Models

* Baum-Welch reestimation algorithm, scaled forward-backward algorithm, Viterbi algorithm
* Support for Input-Output Hidden Markov Models
* Write your own output or transition probability distribution or use the provided distributions, including neural network based conditional probability distributions
* Neural Networks

### Feed-forward backpropagation neural networks of arbitrary topology
* Configurable error functions with sum of squares, weighted sum of squares
* Multiple activation functions with logistic sigmoid, linear, tanh, and soft max
* Choose your weight update rule with standard update rule, standard update rule with momentum, Quickprop, RPROP
* Online and batch training
* Support Vector Machines

### Fast training with the sequential minimal optimization algorithm
* Support for linear, polynomial, tanh, radial basis function kernels
* Decision Trees

### Information gain or GINI index split criteria
* Binary or all attribute value splitting
* Chi-square signifigance test pruning with configurable confidence levels
* Boosted decision stumps with AdaBoost
* K Nearest Neighbors

### Fast kd-tree implementation for instance based algorithms of all kinds
* KNN Classifier with weighted or non-weighted classification, customizable distance function
* Linear Algebra Algorithms

### Basic matrix and vector math, a variety of matrix decompositions based on the standard algorithms
* Solve square systems, upper triangular systems, lower triangular systems, least squares
* Singular Value Decomposition, QR Decomposition, LU Decomposition, Schur Decomposition, Symmetric Eigenvalue Decomposition, Cholesky Factorization
* Make your own matrix decomposition with the easy to use Householder Reflection and Givens Rotation classes
* Optimization Algorithms

### Randomized hill climbing, simulated annealing, genetic algorithms, and discrete dependency tree MIMIC
* Make your own crossover functions, mutation functions, neighbor functions, probability distributions, or use the provided ones.
* Optimize the weights of neural networks and solve travelling salesman problems
* Graph Algorithms

### Kruskals MST and DFS
* Clustering Algorithms

### EM with gaussian mixtures, K-means
* Data Preprocessing

### PCA, ICA, LDA, Randomized Projections
* Convert from continuous to discrete, discrete to binary
* Reinforcement Learning

### Value and policy iteration for Markov decision processes

[1]: https://help.github.com/articles/using-pull-requests

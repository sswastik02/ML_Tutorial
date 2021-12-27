# Types of Data
There are 2 kinds of data:
    * Numerical
    * Categorical (non-numerical)

Categorical data doesnt make sense to the model training so it is replaced by numbers, 
for example in a train there is first class second class and general , They can be converted to 0,1,2

# Tensorflow Core Learning Algorithms


## Linear Regression
It involves finding the "line of best fit" (line equally closest to all data points)
<img src = "https://miro.medium.com/max/1400/1*Cw5ZSYDkIFpmhBwr-hN84A.png" alt="https://miro.medium.com/max/1400/1*Cw5ZSYDkIFpmhBwr-hN84A.png">

So for specefic x you can predict a y using linear regression
It is used when data-points are linearly related

## Training Process

### 1. Batches
To feed a huge amount of data to a model (like 25TB) you have to break them into smaller parts and then feed them.
Smaller parts are known as batches (usually a size of 32 is considered)

These batches are fed to the model multiple times depending on the number of epochs

### 2. Epochs 
Number of epochs we define is the amount of times our model will see the entire dataset
Ex. If we have 10 epochs, our model will see the same dataset 10 times

### 3. Input Function
To pass these datasets we will need an input function to actually input the data into the model

## Classification

It is seperating datapoints into different categories rather than predicting a value such as chance of survival

#### DNN Classifier is recommended over Linear classifier in tensorflow

## Clustering

Clustering is basically grouping of simmilar data

### K - Means Clustering

Let data points be plottable on a graph paper, then algorithm to make clusters is - 
1. Choose k points (on the graph), and assign them as centroids
2. Group each datapoint under one of these centroids 
3. Move the centroid to the center of these groups and regroup all the data points
4. Repeat till all the data points do not change their groups
5. Then you would have got k clusters

## Hidden Markov Models
The Hidden Markov Model is a finite set of states, each of which is associated with a generally multidimensional probability distribution
Transitions among states are governed by a set of probablilities known as transition probabilities

It works with probabilities to predict future events or states.

Elements of Markov Model:

States : Each model has a finite amount of states. Some examples of states in a weather model are sunny and cloudy (2 states here)
            These are hidden, we do not directly observe them

Observations : Each state has a particular outcome or observation based on probability distribution.For example,  If it's a sunny day there
                is a 20% chance of catching a cold, while if it's a cloudy day there is a 70% chance of catching a 
                
Transitions : Each state has a probability of changing to a different state. For example on a sunny day, the chance of having a cloudy day
                the next day is 30% while if its a cloudy day there is a 40% chance next day will be a sunny day

<img src = "./markov.png" alt = "States Image">

## Neural Networks


### 1. Input Layer

This layer takes in raw input for the neural network, To input n peices of information, we need n input neurons

### 2. Output Layer

This layer gives out output from the neural network. For each output information we need 1 output neuron
Say we were classifying, we could do it using one output or probabilities of all classes    

### 3. Hidden Layer

This layer is where all the computation takes place, the reason it's called hidden is because we don't know what's going on in here
You can have multiple layers under hidden layer

### 4. Weights

Every Layer is connected to another layer with connections known as weights
Weights represented as numbers. Generally these numbers are between 0 and 1, But we could have negative numbers, or large numbers.
If all the neurons in the current layer are connected to every neuron in the previous layer`, it is said to be a densly connected layer

These weights are what is changed by the neural network as it goes through datasets to give most accurate outputs

Weight is a trainable parameter

### 5. Biases

There exists only one bias per layer and it affects the next layer
The weights that connect the bias to the layer has a numeric value of 1

Bias is a trainable parameter


<img src="./neural.png" alt = "Neural Network">

## Activation Functions

Now it is highly possible that the output layer of the neural network gives a very large number given so many weights and biases
So to optimise this value Activation Functions are used

The whole point of activation function is to introduce complexity

### 1. Relu (Rectified Linear Unit)
Any Value less than 0 is made 0 and all others are as it is

<img src="https://i.imgur.com/gKA4kA9.jpg">

### 2. Tanh (Hyperbolic Tangent)

It squishes the numbers to fit between -1 and 1

<img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2017/01/tanh.png?resize=456%2C300&ssl=1">

### 3. Sigmoid Function

It squishes values to between 0 and 1

<img src="https://miro.medium.com/max/1838/1*a04iKNbchayCAJ7-0QlesA.png" width="550px">

They move data points to a different plane if all data points are on a single plane

Let's Say it(datapoints) forms a rectange (length and width) and after applying activation functions it becomes a cube (length, height, width, no. of faces, uniformity, etc.) this allows to find patterns that were not present before.


## Training Neural Networks

### 1. Loss function : 
This function determines while training how far the the output value is from the actual value and tells in which direction or how the 
neural network is supposed to change. For example it could indicate that a drastic change in weights and biases is needed or maybe a little change. We need the network loss function to output the least amount of loss

Some loss functions are : 
* Mean Squared Error
* Mean Absolute Error
* Hinge Loss

### 2. Gradient Descent : 
This determines in which direction to move so that the loss from the loss function can be minimised and there is an algorithm called 
back propagation which goes layer by layer backwards from the output neuron to adjust the weights and biases to improve the prediction
of the neural network

<img src="./grad-dec.png" width="500px">

### 3. Optimizer :
It is basically the function that implements backpropagation algorithm

  1. Gradient descent
  2. Stochastic gradient descent
  3. Mini-batch gradient descent
  4. Momentum
  5. Nesterov Accelerated Gradient




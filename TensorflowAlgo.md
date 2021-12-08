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

### Batches
To feed a huge amount of data to a model (like 25TB) you have to break them into smaller parts and then feed them.
Smaller parts are known as batches (usually a size of 32 is considered)

These batches are fed to the model multiple times depending on the number of epochs

### Epochs 
Number of epochs we define is the amount of times our model will see the entire dataset
Ex. If we have 10 epochs, our model will see the same dataset 10 times

### Input Function
To pass these datasets we will need an input function to actually input the data into the model

## Classification

It is seperating datapoints into different categories rather than predicting a value such as chance of survival

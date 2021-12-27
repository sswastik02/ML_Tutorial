import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------- Datasets ---------------------------------

# data sets used here is from mnist fashion dataset from keras (it is preloaded in keras)
fashion_mnist = tf.keras.datasets.fashion_mnist
data = fashion_mnist.load_data()
# print(data) =======> huge array

# print(type(data)) =========> tuple
# print(len(data)) =========> 2

#  ==================================================> (a,b)

# print(type(data[0])) ======> tuple
# print(len(data[0])) =========> 2

# ====================================================> ((a,b),(c,d))

# print(type(data[0][0])) =====> numpy.ndarray

# print(len(data[0][0])) ========> 60000  ==============================> Training dataset (data[0])

# print(data[0][1])
# print(len(data[0][1]))
# print(min(data[0][1]))
# print(max(data[0][1]))
# ======================================> they range from 0 to 9


# print(len(data[1][0])) ==========> 10000 =============================> Testing dataset  (data[1])

# print(data[1][1])
# print(len(data[1][1]))
# print(min(data[1][1]))
# print(max(data[1][1]))

# =====================================> they range from 0 to 9

# The value that ranges from 0 to 9 is probably classifications as for every image there is one value (10 classifications)


# print(type(data[0][0][0])) =======> numpy.ndarray
# print(len(data[0][0][0])) ==========> 28

# print(type(data[0][0][0][0])) =======> numpy.ndarray
# print(len(data[0][0][0][0])) ==========> 28
 
# print(type(data[0][0][0][0][0]))  =======> numpy.uint8

# =======================================================>  Contains data of the form 28 by 28 integers

(train_images, train_labels),(test_images,test_labels) = data

# train_images contain 60000 sets of 28 by 28 integers ( we know that they are values of grayscale images)

# To visualize the images

# def visualize_image(index):
#     plt.figure()
#     plt.imshow(train_images[index],cmap="gray")
#     plt.show()

# for i in range(10):
#     visualize_image(np.where(train_labels == i)[0][0])

# from the above you can conclude the class names

class_names = ["T-shirt",'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# Its not neccessary to give names as you can just work with numbers


# ------------------------------------------ Data PreProcesseing ----------------------------------------------------

train_images = train_images/255.0
test_images = test_images/255.0

# The reason to do this is to keep the value between 0 and 1 as pixels vary between 0 and 255 this way 
# while balancing the weights and biases of the neural network the change is not as much
# so there is a more precise changing of weights and biases giving accurate predictions

# ------------------------------------------ Building Neural Network Model -------------------------------------------
 
model = tf.keras.Sequential([ # Sequential Neural Network (the default one)
    tf.keras.layers.Flatten(input_shape=(28,28)), # Input Layer (Flatten basically linearises the 28 by 28 data (so 784 input neurons))
    tf.keras.layers.Dense(128,activation="relu"), # 1st Hidden Layers (128 is chosen randomly ) (its a dense layer all neurons are connected to all the neurons of previous layer)
    tf.keras.layers.Dense(10,activation="softmax") #  Output layer (10 is chosen because there are 10 classes) ("softmax gives probabilies for each class")
])

# ------------------------------------------- Compiling Neural Network Model----------------------------------------------

model.compile(
    optimizer = "adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'],
)

# Contanst terms such as adam and relu present in building and compiling along with other constant strings are called hyper parameters
# which can be tuned to improve the model. This is known as hyper parameter tuning  

# ------------------------------------------- Training the neural network model ---------------------------------------------

model.fit(train_images,train_labels, epochs=2)
# 2 provided the best result
# ---------------------------------------------------- Testing the model -----------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_images,test_labels,verbose=1)

print(test_accuracy)

# you might notice that test accuracy with training dataset is higher than testing dataset ........... this is called ovefitting the data
# the model has kind of memorised the training dataset so it is also not able to generalise properly


# ----------------------------------------------------- Predicting ---------------------------------------------------------------


def predict():
    predictions = model.predict(test_images)

    while True:
        ele = int((input("Enter a Number: ")))
        if ele < 0 or ele >= 10000:
            print("Invalid\n")
            break
        print("Expected: ",class_names[test_labels[ele]])
        plt.figure()
        plt.imshow(test_images[ele],cmap="gray")
        plt.show()
        print("Predicted: ",class_names[np.argmax(predictions[ele])])

predict()




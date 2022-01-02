import imp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models,layers
# ---------------------------------------------------------- Datasets ---------------------------------------------------

datasets = tf.keras.datasets.cifar10.load_data()

# print(type(datasets)) =========> tuple
# print(len(datasets)) ===========> 2

# print(type(datasets[0])) ======> tuple
# print(len(datasets[0])) =======> 2

# print(type(datasets[1])) =======> tuple
# print(len(datasets[1])) ========> 2

# print(type(datasets[0][0]))
# print(type(datasets[0][1]))
# print(type(datasets[1][0]))
# print(type(datasets[1][1]))
#  =============================> numpy.ndarray



# print(len(datasets[0][0]))
# print(len(datasets[0][1]))
# ===================================> 50000 (training images)

# print(len(datasets[1][0]))
# print(len(datasets[1][1]))

# ===================================> 10000 (testing images)

# print(type(datasets[0][0][0])) =============================> numpy.ndarray
# print(len(datasets[0][0][0])) =============================> 32

# print(type(datasets[0][0][0][0])) =============================> numpy.ndarray
# print(len(datasets[0][0][0][0])) =============================> 32

# print(type(datasets[0][0][0][0][0])) =============================> numpy.ndarray
# print(len(datasets[0][0][0][0][0])) =============================> 3

# print(type(datasets[0][0][0][0][0][0])) =============================> numpy.uint8


# 32x32 images with (rgb)


# print(max(datasets[0][1])[0],min(datasets[0][1])[0]) ===========> 9 0 (10 classes)

(train_images,train_labels),(test_images,test_lables) = datasets



# for i in range(10):
#     plt.imshow(train_images[np.where(train_labels == i)[0][0]])
#     plt.show()

class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Noralisation of data 

train_images = train_images / 255.0 
test_images = test_images / 255.0

# -------------------------------------------------- CNN Architecture -------------------------------------------------

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (32,32,3))) # 32 filters of 3x3, ouptut_shape = (30,30,32) *no padding
model.add(layers.MaxPooling2D(2,2)) # for every 2x2 grid in the image, output_shape = (15,15,32)
model.add(layers.Conv2D(64,(3,3),activation='relu')) # output_shape = (13,13,64)
model.add(layers.MaxPooling2D(2,2)) # output_shape = (6,6,64)
model.add(layers.Conv2D(64,(3,3),activation='relu')) # output_shape = (4,4,64)

# ===================================================== Dense Layers =========================================================

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10)) # amount of classes
model.summary()


#  ---------------------------------------------------------- Compiling the model --------------------------------------

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'],)

# ------------------------------------------------------- Training the model ---------------------------------------------


model.fit(train_images,train_labels,epochs=4,validation_data=(train_images,train_labels))

# ----------------------------------------------------------Testing the model ---------------------------------------------

test_loss, test_acc = model.evaluate(test_images,test_lables,verbose=2)
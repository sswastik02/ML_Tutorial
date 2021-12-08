from pandas.core.indexes import base
import tensorflow as tf
import pandas as pd

# ------------------------------------- Classification -------------------------------------- 


# =================================== Data Sets =====================================================
flower_train_path = tf.keras.utils.get_file(
    "iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
flower_test_path = tf.keras.utils.get_file(
    "iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

# using keras here

CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv(flower_train_path,names=CSV_COLUMN_NAMES,header=0)
test = pd.read_csv(flower_test_path,names=CSV_COLUMN_NAMES,header=0)
# print(train.head(20)) 
# first line revealed the SHAPE OF MODEL and name of species so excluding 1st line using header = 0
# All the data here is numerical

spec_train = train.pop('Species')
spec_test = test.pop('Species')


def input_fn(features,labels,training = True,batch_size = 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if training :
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

feature_columns = []

for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key))

print(feature_columns)

# ---------------------------------------- Building the Model --------------------------------------------------

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=3, # 3 classes to classify into
    hidden_units=[30,10] # 30 in the first hidden layer and 10 in the second hidden layer
)

# classifier.train(
#     input_fn=lambda : input_fn(train,spec_train),
#     steps=5000 # this is a lot like epochs, instead of going through the entire dataset in terms of epochs, we 
#     # see to it that steps number of datapoints are read by the model
# )

# You don't have to train the classifier everytime you run the program, uncomment once to train

test_result = classifier.evaluate(input_fn=lambda : input_fn(test,spec_test,training=False))
print("Accuracy:",test_result['accuracy'])

# ------------------------------------------ Prediction using classifiers-----------------------------

def predict_input_fn(features,batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

input_dict = {}
print("Enter Value as prompted")
# removing species from feature
for feature in CSV_COLUMN_NAMES[:-1]:
    valid = True
    while valid:
        val = input(feature+" : ")
        if not val.isdigit():
            valid = False
    input_dict[feature] = [float(val)] # tensorflow expects the prediction in form of a list, that way you can make multiple predictions in on go

prediction = classifier.predict(lambda:predict_input_fn(input_dict))

for pred_dict in prediction:
    # print(pred_dict)
    species_id = pred_dict['class_ids'][0] # Gives the id of the category which is most probable
    species = SPECIES[species_id]
    chance = pred_dict['probabilities'][species_id]

    print("Prediction:",species,"\nChance:",chance)

# 5.1, 3.3, 1.7, 0.5 ============> Setosa
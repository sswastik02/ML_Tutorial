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

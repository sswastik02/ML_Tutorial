from pandas import *
import matplotlib.pyplot as plt
import tensorflow as tf

# --------------------- Data Sets --------------------------------------------------

dftrain = read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data set
dfeval = read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data set


# print(dftrain.head()) # prints first 5 values unless specified otherwise in the parameter
# the above line returns a dataframe which is a well organised format after importing from csv file

# now we will seperate the survived list from the rest of the dataset ..... we will create the algorithm to predict chances of survival

s_train = dftrain.pop('survived')
s_eval = dfeval.pop('survived')

# Uncomment the below block for seeing graphs and values

# print(dftrain.head())
# print(dftrain.loc[0],dftrain['age']) # different applications of dataframe
# print(dftrain.describe()) # gives statistical analysis of model
# print(dftrain.shape) # like tensors even dataframes have shapes
# dftrain.age.hist(bins=20)
# plt.show() # this is necessary for terminal execution and display of histogram
# dftrain.sex.value_counts().plot(kind = 'barh')
# plt.show()
# dftrain['class'].value_counts().plot(kind = "barh")
# plt.show()
# concat([dftrain,s_train],axis='columns').groupby('sex').survived.mean().plot(kind = "barh")
#  # this will take the concatenation of survials and rest of the dataset and then group them by sex then find mean of survived which is either 0 or 1
# plt.show()

# After seeing the above graphs there are conclusions 
# 1. Majority of passengers are in their 20s or 30s
# 2. Majority of passenges are male
# 3. Majority of passenges are in third class
# 4. Females have higher chance of survival

# -------------------------- Data Conversion ----------------------------------------

categorical_columns = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone'] # dont have to name all of them but its good to
numerical_columns = ['age','fare']

feature_columns=[]
for feature in categorical_columns:
    vocabulary_list = dftrain[feature].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature,vocabulary_list))
    # the argument inside append actually returns a column of a feature containing vocabulary_list(it is coverted to IDs )(red : 1,blue:2)

for feature in numerical_columns:
    vocabulary_list = dftrain[feature].unique()
    feature_columns.append(tf.feature_column.numeric_column(feature,dtype=tf.float32))

# feature_columns now contains each column in a numerical format


#------------------------- The Training Process --------------------------------

def make_input_fn(data_df,label_df,num_epochs = 10,shuffle = True,batch_size = 32):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df)) # label_df contains the metadata while data_df contains data
        # above line returns a dataset we can now use
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds # returns a batch of dataset
    return input_fn

# make_input_fn makes an input function for a particular data frame and label data frame

train_input_fn = make_input_fn(dftrain,s_train)
eval_input_fn = make_input_fn(dfeval,s_eval,num_epochs=1,shuffle=False)
# No need to send same dataset twice or shuffle during evaluating

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# Estimator is the implementation of ML Algotithms in tensorflow, so the above lines create a ML model

# Now to train the model

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn) # To evaluate the model after the training

print(result['accuracy']) # result variable contains more statistical data besides accuracy

# To predict for a value
predict = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[2])
print(s_eval.loc[2])
print("Survival Chance:",predict[2]["probabilities"][1]) # since death is shown by survival 0 and survived by survival 1 [1] will indicate survival






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
    feature_columns.append(tf.feature_column.numeric_column(feature,vocabulary_list))

# feature_columns now contains each column in a numerical format








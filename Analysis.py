import numpy as np
import pandas as pd 

#importing test and train dataset as dataframe
Train = pd.DataFrame(pd.read_csv('train.csv'))
Test = pd.DataFrame(pd.read_csv('test.csv'))

#Checking dataset values 
print(Train.head())
print(Test.head())

#Seeing number of rows and columns
print(Train.shape)
print(Test.shape)

#checking dataset details 
print(Train.describe())
print(Test.describe())

#Converting string values to float for processing 

"""
#Removing constant columns (0 variance)
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(Train)
len(Train.columns[constant_filter.get_support()])
constant_columns = [column for column in Train.columns
                    if column not in Train.columns[constant_filter.get_support()]]

print(len(constant_columns))
for column in constant_columns:
    print(column)

Train = constant_filter.transform(Train)
Test = constant_filter.transform(Test)

print(Train.describe())
print(Test.describe())

#Remove columns with 0 variance 
Train = Train.loc[:, (Train != Train.iloc[0]).any()] 
print(Train.head())

Test = Test.loc[:, (Test != Test.iloc[0]).any()] 
print(Test.head())
"""
#Checking for null and unique values 
print(Train.isnull())
print(Train.nunique(axis=1))
print(Test.isnull())
print(Test.nunique(axis=1))

# Identify your target variable

X = Train.drop(columns=['ID', 'y'])
y = Train['y']

new_data = Test.drop(columns=['ID'])

# Split df in train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

#Apply Encoder 
#from feature_engine.categorical_encoders import OrdinalCategoricalEncoder, RareLabelCategoricalEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# Tranform X_train, X_test and new_data

X_train = X_train.apply(le.fit_transform)
X_test = X_test.apply(le.fit_transform)
new_data = new_data.apply(le.fit_transform)


#Performing dimensional reduction 
from sklearn.decomposition import PCA

# define your object preserving 90% variance

pca = PCA(n_components=0.90)

# fit your training data

pca.fit(X_train)

# Tranform X_train, X_test and new_data

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
new_data = pca.transform(new_data)


#Performing XGBoost 
from xgboost import XGBRegressor

# define your object using default hyperameters

xgbr = XGBRegressor()

# fit your training data

xgbr.fit(X_train, y_train)

#Performance matrix 
from sklearn.metrics import mean_squared_error
import math

y_preds =  xgbr.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_preds))
print(rmse)

# Predicting Test Data 
project_preds = xgbr.predict(new_data)
print(project_preds)

# Save a pickel file 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(xgbr,pickle_out)
pickle_out.close()

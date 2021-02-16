# MercDataAnalysisModel
Prediction of Merc Data set using XGBoost

# DESCRIPTION

Reduce the time a Mercedes-Benz spends on the test bench.

Problem Statement Scenario:
Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.

To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.

You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.

Following actions should be performed:

If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
Check for null and unique values for test and train sets.
Apply label encoder.
Perform dimensionality reduction.
Predict your test_df values using XGBoost.

# How to install 
Use command pip install -r requirements.txt to add all the dependencies and run Analysis.py 

# Project Step by Step 
1. If for any column(s), the variance is equal to zero, then you need to remove those variable(s).

a. Data Importing and checking : 
Code : 
import numpy as np
import pandas as pd 

#importing test and train dataset as dataframe
Train = pd.DataFrame(pd.read_csv('train.csv'))
Test = pd.DataFrame(pd.read_csv('test.csv'))

#Checking dataset values 
print(Train.head())
print(Test.head())

#Seeing number of rows and columns
print(Train.shape)
print(Test.shape)

#checking dataset details 
print(Train.describe())
print(Test.describe())

Output : 

(env) C:\Users\Anirudh\Desktop\Anirudh\DataScience\Git Projects\MercDataAnalysis>python Analysis.py

   ID       y  X0 X1  X2 X3 X4 X5 X6 X8  X10  ...  X374  X375  X376  X377  X378  X379  X380  X382  X383  X384  X385
   
0   0  130.81   k  v  at  a  d  u  j  o    0  ...     0     0     0     1     0     0     0     0     0     0     0  

1   6   88.53   k  t  av  e  d  y  l  o    0  ...     0     1     0     0     0     0     0     0     0     0     0  

2   7   76.26  az  w   n  c  d  x  j  x    0  ...     0     0     0     0     0     0     0     1     0     0     0  

3   9   80.62  az  t   n  f  d  x  l  e    0  ...     0     0     0     0     0     0     0     0     0     0     0  

4  13   78.02  az  v   n  f  d  h  d  n    0  ...     0     0     0     0     0     0     0     0     0     0     0  

[5 rows x 378 columns]
   ID  X0 X1  X2 X3 X4 X5 X6 X8  X10  X11  ...  X374  X375  X376  X377  X378  X379  X380  X382  X383  X384  X385
   
0   1  az  v   n  f  d  t  a  w    0    0  ...     0     0     0     0     1     0     0     0     0     0     0    

1   2   t  b  ai  a  d  b  g  y    0    0  ...     0     0     0     1     0     0     0     0     0     0     0     

2   3  az  v  as  f  d  a  j  j    0    0  ...     0     0     0     0     1     0     0     0     0     0     0     

3   4  az  l   n  f  d  z  l  n    0    0  ...     0     0     0     0     1     0     0     0     0     0     0     

4   5   w  s  as  c  d  y  i  m    0    0  ...     0     1     0     0     0     0     0     0     0     0     0     

[5 rows x 377 columns]
(4209, 378)

(4209, 377)

                ID            y          X10     X11  ...         X382         X383         X384         X385
                
count  4209.000000  4209.000000  4209.000000  4209.0  ...  4209.000000  4209.000000  4209.000000  4209.000000     

mean   4205.960798   100.669318     0.013305     0.0  ...     0.007603     0.001663     0.000475     0.001426      

std    2437.608688    12.679381     0.114590     0.0  ...     0.086872     0.040752     0.021796     0.037734        

min       0.000000    72.110000     0.000000     0.0  ...     0.000000     0.000000     0.000000     0.000000        

25%    2095.000000    90.820000     0.000000     0.0  ...     0.000000     0.000000     0.000000     0.000000        

50%    4220.000000    99.150000     0.000000     0.0  ...     0.000000     0.000000     0.000000     0.000000        

75%    6314.000000   109.010000     0.000000     0.0  ...     0.000000     0.000000     0.000000     0.000000        

max    8417.000000   265.320000     1.000000     0.0  ...     1.000000     1.000000     1.000000     1.000000        

[8 rows x 370 columns]
                ID          X10          X11          X12  ...         X382         X383         X384         X385
                
count  4209.000000  4209.000000  4209.000000  4209.000000  ...  4209.000000  4209.000000  4209.000000  4209.000000   

mean   4211.039202     0.019007     0.000238     0.074364  ...     0.008791     0.000475     0.000713     0.001663   

std    2423.078926     0.136565     0.015414     0.262394  ...     0.093357     0.021796     0.026691     0.040752   

min       1.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000   

25%    2115.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000   

50%    4202.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000   

75%    6310.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000   

max    8416.000000     1.000000     1.000000     1.000000  ...     1.000000     1.000000     1.000000     1.000000   

[8 rows x 369 columns]


b) Removing data with 0 variance :
Code : 

Train = Train.loc[:, (Train != Train.iloc[0]).any()] 
print(Train.head())

Test = Test.loc[:, (Test != Test.iloc[0]).any()] 
print(Test.head())


Output : 

(env) C:\Users\Anirudh\Desktop\Anirudh\DataScience\Git Projects\MercDataAnalysis>python Analysis.py

   ID       y  X0 X1  X2 X3 X4 X5 X6 X8  X10  ...  X374  X375  X376  X377  X378  X379  X380  X382  X383  X384  X385
   
0   0  130.81   k  v  at  a  d  u  j  o    0  ...     0     0     0     1     0     0     0     0     0     0     0  

1   6   88.53   k  t  av  e  d  y  l  o    0  ...     0     1     0     0     0     0     0     0     0     0     0  

2   7   76.26  az  w   n  c  d  x  j  x    0  ...     0     0     0     0     0     0     0     1     0     0     0  

3   9   80.62  az  t   n  f  d  x  l  e    0  ...     0     0     0     0     0     0     0     0     0     0     0  

4  13   78.02  az  v   n  f  d  h  d  n    0  ...     0     0     0     0     0     0     0     0     0     0     0  

[5 rows x 366 columns]
   ID  X0 X1  X2 X3 X4 X5 X6 X8  X10  X11  ...  X374  X375  X376  X377  X378  X379  X380  X382  X383  X384  X385
   
0   1  az  v   n  f  d  t  a  w    0    0  ...     0     0     0     0     1     0     0     0     0     0     0     

1   2   t  b  ai  a  d  b  g  y    0    0  ...     0     0     0     1     0     0     0     0     0     0     0     

2   3  az  v  as  f  d  a  j  j    0    0  ...     0     0     0     0     1     0     0     0     0     0     0     

3   4  az  l   n  f  d  z  l  n    0    0  ...     0     0     0     0     1     0     0     0     0     0     0     

4   5   w  s  as  c  d  y  i  m    0    0  ...     0     1     0     0     0     0     0     0     0     0     0     

[5 rows x 372 columns]

Screenshots : 





2. Check for null and unique values for test and train sets
Code : 
#Checking for null and unique values 
print(Train.isnull())
print(Train.nunique(axis=1))
print(Test.isnull())
print(Test.nunique(axis=1))


Output : 

(env) C:\Users\Anirudh\Desktop\Anirudh\DataScience\Git Projects\MercDataAnalysis>python Analysis.py

         ID      y     X0     X1     X2     X3     X4     X5  ...   X377   X378   X379   X380   X382   X383   X384   
         
X385

0     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

1     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

2     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

3     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

...     ...    ...    ...    ...    ...    ...    ...    ...  ...    ...    ...    ...    ...    ...    ...    ...   

 ...
 
4205  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4206  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4207  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4208  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

[4209 rows x 378 columns]
0       11
1       12
2       11
3       12
4       10
        ..
4204    11
4205    10
4206    12
4207    12
4208    12
Length: 4209, dtype: int64
         ID     X0     X1     X2     X3     X4     X5     X6  ...   X377   X378   X379   X380   X382   X383   X384  
         
X385

0     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

1     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

2     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

3     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4     False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

...     ...    ...    ...    ...    ...    ...    ...    ...  ...    ...    ...    ...    ...    ...    ...    ...   

 ...
 
4204  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4205  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4206  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4207  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

4208  False  False  False  False  False  False  False  False  ...  False  False  False  False  False  False  False  False

[4209 rows x 377 columns]

0       10

1       10

2       10

3        9

4       11

        ..
        
4204    11

4205     9

4206    10

4207    11

4208    10

Length: 4209, dtype: int64


3. Apply label encoder :-

Code : 
# Identify your target variable

X = Train.drop(columns=['ID', 'y'])
y = Train['y']

new_data = Test.drop(columns=['ID'])

#Split df in train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

#Apply Encoder 
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder, RareLabelCategoricalEncoder

rare = RareLabelCategoricalEncoder()
rare.fit(X_train)

# Tranform X_train, X_test and new_data

X_train = rare.transform(X_train)
X_test = rare.transform(X_test)
new_data = rare.transform(new_data)

# define your object
encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary', )

# fit your training data
encoder.fit(X_train)

# Tranform X_train, X_test and new_data

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
new_data = encoder.transform(new_data)


4. Perform dimensionality reduction.

#Performing dimensional reduction 
from sklearn.decomposition import PCA

#define your object preserving 90% variance

pca = PCA(n_components=0.90)

#fit your training data

pca.fit(X_train)

#Tranform X_train, X_test and new_data

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
new_data = pca.transform(new_data)



5. Predict your test_df values using XGBoost :
#Performing XGBoost 
from xgboost import XGBRegressor

#define your object using default hyperameters

xgbr = XGBRegressor()

# fit your training data

xgbr.fit(X_train, y_train)

#Performance matrix 
from sklearn.metrics import mean_squared_error
import math

y_preds =  xgbr.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_preds))
print(rmse)

#Predicting Test Data 
project_preds = xgbr.predict(new_data)
project_preds

Output : 
10.401419251378262

# References 
Features selection(removing constant variance columns )

https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/

https://github.com/Simplilearn-Edu/Machine-Learning--Projects

https://colab.research.google.com/drive/1qRhjzKdPuYAg6XvJpIrvptj4ykQHCPyq

https://www.kaggle.com/eswarchandt/amazon-user-based-recommendation-system


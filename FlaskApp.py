import numpy as np 
import pandas as pd 
import pickle

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

df_test = pd.read_csv('test.csv')
prediction = classifier.predict(df_test)
print(prediction)

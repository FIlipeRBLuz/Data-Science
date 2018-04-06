# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:36:50 2018

@author: filipe.luz
"""

#svr

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
'''As the dataset is short we don't need to split it 
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
#SVR don't apply scaling, so we need do it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Predicting a new result 
'''as we need find one value, let's put the exactly value without 
the matrix X'''
'''After transform the scale we need adjust all values that we need to predict to the same 
datatype scale'''
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
'''How we need see the original we need to inverse the method'''



#Visualising the SVR results
plt.scatter(X, y, color ='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(SVR )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualising the SVR results
plt.scatter(X, y, color ='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(SVR )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
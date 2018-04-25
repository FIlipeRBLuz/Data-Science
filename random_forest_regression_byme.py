# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:32:54 2018

@author: filipe.luz
"""

#Random Forest Regression

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
#Not necessary here because most of algorithms will trate it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,
                                  random_state = 0)

regressor.fit(X, y)



#Prdicting a new result 
'''as we need find one value, let's put the exactly value without 
the matrix X'''
y_pred = regressor.predict(6.5)


#Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color ='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

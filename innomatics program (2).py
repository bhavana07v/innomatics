#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[26]:


df = pd.read_csv(r'C:\Users\KUMAR\Downloads\dataframe_.csv')


# In[24]:


# Plot the data to visualize the distribution and check for outliers
plt.scatter(df['input'], df['output'])
plt.xlabel('input')
plt.ylabel('output')
plt.show()


# In[19]:


# Detect outliers using the Z-score method
z_scores = (df - df.mean()) / df.std()
df = df[(z_scores < 3).all(axis=1)]


# In[20]:


# Feature engineering - calculate BMI
df['bmi'] = df['output'] / (df['input']/12)**2


# In[21]:


# Split the data into training and test sets
X = df[['input', 'bmi']]
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[22]:


# Train a linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)


# In[23]:


# Hyperparameter tuning - try different regularization parameters
from sklearn.linear_model import Ridge
alphas = [0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error for alpha =', alpha, ':', mse)


# In[ ]:





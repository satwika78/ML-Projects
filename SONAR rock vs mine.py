#!/usr/bin/env python
# coding: utf-8

# # SONAR ROCK AND MINE DETECTION 

# ***Importing required libraries***

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# ***Pre-Processing the dataset***

# In[2]:


data = pd.read_csv('C:/Users/satwika/Downloads/sonar.csv')
data.head()


# In[3]:


target = data['label']
target.head()


# In[4]:


data = data.drop('label',axis=1)
data.head()


# In[5]:


data.shape


# Feature Selection

# In[78]:


clf=RandomForestClassifier(n_estimators=10)
clf.fit(data,target)


# In[79]:


clf.feature_importances_


# In[80]:


model = SelectFromModel(clf,prefit=True)
data_new = model.transform(data)
data_new.shape


# In[81]:


from sklearn.model_selection import train_test_split


# In[93]:


X_train,X_test,Y_train,Y_test = train_test_split(data_new,target,test_size=0.1)


# # ***Training the model***

# ***Logistic Regression***

# In[94]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# In[95]:


model.score(X_test,Y_test)


# ***Support Vector Machine***

# In[96]:


from sklearn.svm import SVC
model1 = SVC(kernel = 'linear', C = 4, gamma = 'scale')


# In[97]:


model1.fit(X_train,Y_train)


# In[98]:


model1.score(X_test,Y_test)


# ***Decision Tree***

# In[101]:


from sklearn import tree
model2 = tree.DecisionTreeClassifier()
model2.fit(X_train,Y_train)


# In[102]:


model2.score(X_test,Y_test)


# ***Making a predictive system***

# In[117]:


input_data = (3.110e-02, 4.910e-02, 8.310e-02, 7.900e-03, 2.000e-02, 1.016e-01,
        2.025e-01, 1.767e-01, 2.555e-01, 2.722e-01, 8.701e-01, 7.672e-01,
        4.148e-01, 8.049e-01, 5.830e-01, 4.676e-01, 3.150e-01, 2.139e-01,
        6.150e-02, 7.790e-02, 8.450e-02, 5.920e-02, 8.900e-03, 1.880e-02)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#predicting using models
prediction = model.predict(input_data_reshaped)
prediction1 = model1.predict(input_data_reshaped)
prediction2 = model2.predict(input_data_reshaped)
X_new = np.array([prediction,prediction1,prediction2])
print(prediction,prediction1,prediction2)

#printing output
R=0
M=0
for i in range(0,3):
    if(X_new[i]=='R'):
        R=R+1
    else:
        M=M+1
        
if (R>M):
  print('The Given Sample is a Rock')
else:
  print('The Given Sample is a Mine')


# In[116]:


input_data = (9.680e-02, 8.210e-02, 6.080e-02, 6.170e-02, 1.207e-01, 4.223e-01,
        5.744e-01, 3.488e-01, 1.700e-01, 3.087e-01, 8.321e-01, 1.000e+00,
        9.600e-02, 3.451e-01, 2.316e-01, 7.375e-01, 7.792e-01, 6.788e-01,
        4.221e-01, 3.067e-01, 1.349e-01, 1.057e-01, 2.060e-02, 1.900e-02)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#predicting using models
prediction = model.predict(input_data_reshaped)
prediction1 = model1.predict(input_data_reshaped)
prediction2 = model2.predict(input_data_reshaped)
X_new = np.array([prediction,prediction1,prediction2])
print(prediction,prediction1,prediction2)

#printing output
R=0
M=0
for i in range(0,3):
    if(X_new[i]=='R'):
        R=R+1
    else:
        M=M+1
        
if (R>M):
  print('The Given Sample is a Rock')
else:
  print('The Given Sample is a Mine')


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # FAKE NEWS DETECTION

# ***importing required libraries***

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.metrics import classification_report
import re #regular expression-for searching words
import string
from nltk.corpus import stopwords #The words which do not add value
from nltk.stem.porter import PorterStemmer #Stem words and gives root words
from sklearn.feature_extraction.text import TfidfVectorizer #text into vectors
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


print(stopwords.words('english'))


# ***Data Pre-Processing***

# In[5]:


news=pd.read_csv('C:/Users/satwika/Documents/Machine Learning/train.csv')
news.head()


# In[6]:


news.shape


# In[7]:


#counting null values
news.isnull().sum()


# In[8]:


#replacing the null values with empty strings
news=news.fillna('')


# In[9]:


#merging author and title
news['content'] = news['author']+' '+news['title']


# In[10]:


news.head()


# In[11]:


#seperating data and labels
X = news.drop(columns='label', axis=1)
Y = news['label']


# In[12]:


print(X)
print(Y)


# In[13]:


#stemming - reducing a word to its root word
port_stem = PorterStemmer()


# In[14]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[15]:


news['content'] = news['content'].apply(stemming)


# In[16]:


print(news['content'])


# In[17]:


#separating the data and label
X = news['content'].values
Y = news['label'].values


# In[18]:


X


# In[19]:


Y


# In[20]:


X.shape


# In[21]:


Y.shape


# In[22]:


# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[23]:


print(X)


# ***splitting the data***

# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# ***Training the data and testing***

# **LOGISTIC REGRESSION**

# In[27]:


model1 = LogisticRegression()
model1.fit(X_train, Y_train)


# In[28]:


X_test_prediction = model1.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score of the Logistic Regression : ',test_data_accuracy)


# **DECISION TREE**

# In[30]:


from sklearn import tree
model2 = tree.DecisionTreeClassifier()
model2.fit(X_train, Y_train)


# In[31]:


X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)
print('Accuracy score of the Decision Tree : ',test_data_accuracy2)


# **RANDOM FOREST**

# In[32]:


from sklearn.ensemble import RandomForestClassifier
model3= RandomForestClassifier(n_estimators= 10, criterion="entropy")
model3.fit(X_train, Y_train)


# In[33]:


X_test_prediction3 = model3.predict(X_test)
test_data_accuracy3 = accuracy_score(X_test_prediction3, Y_test)
print('Accuracy score of the Random Forest : ',test_data_accuracy3)


# **Making a Predictive System**

# In[34]:


def run(num):
    v1=model1.predict(X_test[num])
    v2=model2.predict(X_test[num])
    v3=model3.predict(X_test[num])
    X_new = np.array([v1,v2,v3])
    count_true=0
    count_false=0
    for i in range(0,3):
        if(X_new[i]==0):
            count_true=count_true+1
        else:
            count_false=count_false+1
    if(count_true>count_false):
        print("The news is Real")
    else:
        print("The news is Fake")


# In[35]:


run(8)


# In[36]:


run(3)


# In[37]:


run(12)


# In[38]:


run(9)


# # ANALYSING THE MODELS

# # Confusion matrix

# ***confusion matrix for logistic regression***

# In[39]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[40]:


news=model1.predict(X_test)


# In[41]:


confusion_matrix(Y_test,news)


# In[42]:


mat=plot_confusion_matrix(model1,X_test,Y_test)
mat.ax_.set_title('Confusion Matrix')
plt.xlabel('Predicted label',color='black')
plt.ylabel('True label',color='black')
plt.show()


# ***confusion matrix for decision tree***

# In[43]:


news1=model2.predict(X_test)


# In[44]:


confusion_matrix(Y_test,news1)


# In[45]:


mat1=plot_confusion_matrix(model2,X_test,Y_test)
mat1.ax_.set_title('Confusion Matrix')
plt.xlabel('Predicted label',color='black')
plt.ylabel('True label',color='black')
plt.show()


# ***confusion matrix for random forest***

# In[46]:


news2=model3.predict(X_test)


# In[47]:


confusion_matrix(Y_test,news2)


# In[48]:


mat2=plot_confusion_matrix(model3,X_test,Y_test)
mat2.ax_.set_title('Confusion Matrix')
plt.xlabel('Predicted label',color='black')
plt.ylabel('True label',color='black')
plt.show()


# In[ ]:





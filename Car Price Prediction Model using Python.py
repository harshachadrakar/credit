#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[2]:


data = pd.read_csv("C:\\Users\\chakr\\OneDrive\\Desktop\\machine learning\\CarPrice.csv")
data.head()


# In[3]:


##There are 26 columns in this dataset, so it is very important to check whether or not this dataset contains null values before going any further:


# In[4]:


data.isnull().sum()


# In[5]:


##this dataset doesn’t have any null values, now let’s look at some of the other important insights to get an idea of what kind of data we’re dealing with:


# In[6]:


data.info()


# In[7]:


print(data.describe())


# In[8]:


data.CarName.unique()


# In[9]:


###The price column in this dataset is supposed to be the column whose values we need to predict. So let’s see the distribution of the values of the price column:


# In[10]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()


# In[11]:


print(data.corr())


# In[12]:


plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[13]:


##Training a Car Price Prediction Model
##I will use the decision tree regression 
###algorithm to train a car price prediction model.So let’s split 
###the data into training and test sets and use the decision tree regression 
###algorithm to train the model:


# In[14]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


# In[15]:


x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[16]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[17]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[18]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Akash Real Estate- Price Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv('data.csv')


# In[3]:


housing.head()


# In[4]:


housing.describe()


# In[5]:


housing.isna().count()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


# data.hist(bins=50 ,figsize=(12,15))


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)


# In[12]:


len(test_set)


# In[13]:


len(train_set)


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)


# In[15]:


for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set= housing.loc[test_index]


# In[16]:


strat_test_set.describe()


# In[17]:


housing= strat_train_set.copy()


# In[18]:


housing_data = strat_train_set.drop('MEDV' , axis =1)
housing_label = strat_train_set['MEDV'].copy()


# In[19]:


from pandas.plotting import scatter_matrix



# In[20]:


attributes = ["MEDV","RM","ZN", "LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[21]:


housing["TAXRM"] = housing["TAX"]/housing["RM"]


# In[22]:


housing.head()


# In[23]:


corr_matrix =housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing.plot(kind = "scatter" , x ="TAXRM" , y= "MEDV" ,alpha=0.8)


# In[25]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[26]:


imputer.statistics_


# In[27]:


X= imputer.transform(housing)


# In[28]:


housing_tr = pd.DataFrame(X,columns = housing.columns)


# In[29]:


housing_tr.describe()


# # features scaling
# 

# In[30]:


# min-max scalling
# sklearn provide min max scalar ----(val-min)/(max-min)
# Standardization
# sklearn provides a class called Standard_Scalar for this


# In[ ]:





# # creating Pipeline

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline =Pipeline([
    ('imputer ',SimpleImputer(strategy="median")),
    ('std_scalar ,', StandardScaler()),
])


# In[32]:


housing_num_tr = my_pipeline.fit_transform(housing_data)


# In[33]:


housing_num_tr


# # seleting a desired model for dragon Estate

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
model = LinearRegression()
# model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_label)


# In[35]:


some_data = housing_data.iloc[:5]
some_label = housing_label.iloc[:5]


# In[36]:


prepare_data = my_pipeline.transform(some_data)


# In[37]:


model.predict(prepare_data)


# In[38]:


some_label


# In[39]:


from sklearn.metrics import mean_squared_error
housing_prediction = model.predict(housing_num_tr)
mse = mean_squared_error(housing_label , housing_prediction)
rmse =np.sqrt(mse)


# In[40]:


mse


# In[41]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(model,housing_num_tr , housing_tr, scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-score)


# In[42]:


rmse_scores


# In[43]:


from joblib import dump,load
dump(model,'Akash.joblib')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Lasso without new features

# In[1]:


# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression


# In[2]:


# Both features and target have already been scaled: mean = 0; SD = 1
data = pd.read_csv('Houseprice_data_scaled.csv') 
data_orig=pd.read_csv('original data.csv') 


# In[3]:


# First 1800 data items are training set; the next 600 are the validation set
train = data.iloc[:1800] 
val = data.iloc[1800:2400]


# In[4]:


# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_val = train.drop('Sale Price', axis=1), val.drop('Sale Price', axis=1)
y_train, y_val = train[['Sale Price']], val[['Sale Price']] 


# In[5]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.05)
model=lasso.fit(X_train, y_train)

coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs
yhat = pd.DataFrame(model.predict(X_val),columns=['yhat'])
mse(yhat,y_val)


# ## Lasso with 2 features

# In[6]:


# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression


# In[7]:


# Both features and target have already been scaled: mean = 0; SD = 1
data = pd.read_csv('Houseprice_data_scaled.csv') 
data_orig=pd.read_csv('original data.csv') 


# In[8]:


data['LotFrontage']=data_orig['LotFrontage']


# In[9]:


#using Multivariate feature imputation
from sklearn import impute
from sklearn import experimental
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data_new=data.drop(columns=['Sale Price'])
lr=LinearRegression()
imp = IterativeImputer( max_iter=10, verbose=2,imputation_order='roman')
a=imp.fit_transform(data_new)


# In[10]:


b=data_new.columns
data_new=pd.DataFrame(a, columns=b)


# In[11]:


# replace with median
data_new['Lot_median']=data_orig.LotFrontage.fillna(data_orig.LotFrontage.median())
data_new.Lot_median.isnull().sum()


# In[12]:


# replace with mean
data_new['Lot_mean']=data_orig.LotFrontage.fillna(data_orig.LotFrontage.mean())
data_new.Lot_mean.isnull().sum()


# In[13]:


data_new['Sale Price']=data['Sale Price']


# In[14]:


from sklearn.linear_model import Lasso
from sklearn import preprocessing

data_with_rep= data_new.drop(columns=['Lot_mean','Lot_median'])
lotshape=pd.get_dummies(data_orig.LotShape)

standard_lotshape = pd.DataFrame(preprocessing.scale(lotshape),columns=lotshape.columns)
print(standard_lotshape.mean())

data_with_rep[['LotShape IR1','LotShape IR2','LotShape IR3','LotShape Reg']]=standard_lotshape


# In[15]:


train_1 = data_with_rep.iloc[:1800] 
val_1 = data_with_rep.iloc[1800:2400]

X_train_1, X_val_1 = train_1.drop('Sale Price', axis=1), val_1.drop('Sale Price', axis=1)
y_train_1, y_val_1 = train_1[['Sale Price']], val_1[['Sale Price']] 


# In[16]:


from sklearn import preprocessing

lasso = Lasso(alpha=0.05)
model=lasso.fit(X_train_1, y_train_1)

coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train_1.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs

yhat_1 = pd.DataFrame(model.predict(X_val_1),columns=['yhat'])
mse(yhat_1,y_val_1)


# In[17]:


np.sqrt(mse(yhat_1,y_val_1))


# ## four features

# In[18]:


data_dummies_4=pd.get_dummies(data_orig[['Functional','SaleCondition']])
standard_features = pd.DataFrame(preprocessing.scale(data_dummies_4),columns=data_dummies_4.columns)
data_standard_4=data_with_rep.join(standard_features)
data_standard_4_trim=data_standard_4.drop(['LotShape Reg','Functional_Typ','SaleCondition_Normal'],axis = 1)


# In[19]:


train_4 = data_standard_4_trim.iloc[:1800] 
val_4 = data_standard_4_trim.iloc[1800:2400]

X_train_4, X_val_4 = train_4.drop('Sale Price', axis=1), val_4.drop('Sale Price', axis=1)
y_train_4, y_val_4 = train_4[['Sale Price']], val_4[['Sale Price']] 

lasso = Lasso(alpha=0.05)
model=lasso.fit(X_train_4, y_train_4)

coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train_4.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs
yhat_4 = pd.DataFrame(model.predict(X_val_4),columns=['yhat'])
np.sqrt(mse(yhat_4,y_val_4))


# In[20]:


mse(yhat_4,y_val_4)


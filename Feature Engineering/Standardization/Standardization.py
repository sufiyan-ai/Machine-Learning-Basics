#!/usr/bin/env python
# coding: utf-8

# ## Standardization

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('Social_Network_Ads.csv')
df


# In[3]:


df=df.iloc[:,2:]
#df=df.loc[:,['Age','EstimatedSalary','Purchased']]
df.head()


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(df.drop('Purchased',axis=1),df['Purchased'],
                                                       test_size=0.3)
x_train.shape,x_test.shape


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[8]:


x_train_scaled=pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled=pd.DataFrame(x_test_scaled,columns=x_test.columns)


# In[9]:


x_train_scaled


# In[10]:


np.round(x_train.describe(),1)


# In[11]:


np.round(x_train_scaled.describe(),1)


# In[12]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.scatter(x_train['Age'],x_train['EstimatedSalary'],color='blue')
ax1.set_title('Before Scaling')
ax2.scatter(x_train_scaled['Age'],x_train_scaled['EstimatedSalary'],color='red')
ax2.set_title('After Scaling')


# In[13]:


fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))
sns.kdeplot(x_train['Age'],color='blue',ax=ax1)
sns.kdeplot(x_train['EstimatedSalary'],ax=ax1,color='r')
ax1.set_title('Before Scaling')

#AFTER SCALING
sns.kdeplot(x_train_scaled['Age'],color='blue',ax=ax2)
sns.kdeplot(x_train_scaled['EstimatedSalary'],color='red',ax=ax2)
ax2.set_title('After Scaling')
plt.show()


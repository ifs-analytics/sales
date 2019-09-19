#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[69]:


df = pd.read_csv('k clustering serials2.csv')
df.columns


# In[148]:


booked = df.groupby(['Partner Serials'])['Loan Booked'].count()
received = df.groupby(['Partner Serials'])['Referral Received'].count()
booked2 = df.groupby(['FA Serials'])['Loan Booked'].count()
received2 = df.groupby(['FA Serials'])['Referral Received'].count()


conv1 = booked/received
conv1 = conv1.to_frame().reset_index()

conv2 = booked2/received2
conv2 = conv2.to_frame().reset_index()


# In[173]:


conv1.columns = ['Partner Serials','Conversion Rate(Partner)']
conv2.columns = ['FA Serials','CR2(FA)']


# In[174]:


train = pd.merge(conv1,df,on='Partner Serials')
test = pd.merge(conv1,df,on='Partner Serials')


# In[175]:


train1 = pd.merge(conv2,train,on='FA Serials')
test1 = pd.merge(conv2,test,on='FA Serials')


# In[176]:


train1


# In[180]:


train1.columns=train1.columns.str.replace(' ','_')
test1.columns=test1.columns.str.replace(' ','_')
conv1.columns=conv1.columns.str.replace(' ','_')


# In[182]:


# Fill missing values with mean column values in the train set
train1.fillna(train1.mean(), inplace=True)


# In[183]:


test1.fillna(test1.mean(), inplace=True)


# In[184]:


print(train1.isna().sum())


# In[185]:


print(test1.isna().sum())


# In[188]:


g = sns.FacetGrid(train1, col='CR2(FA)')
g.map(plt.hist, 'Partner_Serials', bins=20)


# In[206]:


train1.info()


# In[204]:


train1 = train1.drop(['App_ID','Partner', 'FA','Lender','State','domains','Finance_Type','Referral_Received','Loan_Booked'], axis=1)
test1 = test1.drop(['App_ID','Partner', 'FA','Lender','State','domains','Finance_Type','Referral_Received','Loan_Booked'] axis=1)


# In[205]:


train1


# In[201]:


train1.fillna(train1.mean(), inplace=True)
test1.fillna(test1.mean(), inplace=True)


# In[207]:


train1 = train1.drop(['Domain_Serials'], axis=1)
test1 = test1.drop(['Domain_Serials'], axis=1)


# In[208]:


X = np.array(train1.drop(['CR2(FA)'], 1).astype(float))


# In[209]:


y = np.array(train1['CR2(FA)'])


# In[214]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[215]:


kmeans.fit(X_scaled)


# In[217]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[ ]:





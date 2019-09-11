#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px


# In[3]:


df = pd.read_csv('models2.csv')


# In[4]:


df


# In[5]:


adr= df[['App ID','Lender','Credit Score','State','Loan Amount']]


# In[6]:


adr1 = adr[adr['Lender'] != 'OpenLending-Air Force']
adr2 = adr[adr['Lender'] != 'Air Force Federal Credit Union']


# In[7]:


adr1 #Air Force Federal Credit Union Dataset


# In[8]:


adr2#open Lending Airforce


# In[9]:


data = adr1
fig1 = px.scatter_3d(data, x='Credit Score',y='State',z='Loan Amount')
fig1.show()


# In[11]:


data2 = adr2
fig2 = px.scatter_3d(data2, x='Credit Score',y='State',z='Loan Amount')
fig2.show()


# In[ ]:





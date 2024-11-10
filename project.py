#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np 


# In[49]:


df1 = pd.read_csv("dataset (1).csv")
df2= pd.read_csv("Language Detection.csv")
df=pd.merge(df1,df2, how='outer')
df


# In[50]:


df.shape


# In[51]:


df['Language'].value_counts()


# In[52]:


import seaborn as sns 
import matplotlib.pyplot as plt 


# In[53]:


plt.figure(figsize=(10,10))
sns.countplot(df['Language'])


# In[54]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Apply the label encoder to the 'Language' column
df['Language'] = label_encoder.fit_transform(df['Language'])



# In[55]:


df


# In[56]:


df.info()


# In[57]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

X = df.drop('Language', axis=1)
y = df['Language']


# In[58]:


X_resampled, y_resampled = rus.fit_resample(X, y)


# In[59]:


X_resampled.value_counts()


# In[60]:


y_resampled


# In[ ]:





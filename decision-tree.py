#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 


# In[4]:


df = pd.read_csv('movie_success_rate.csv')
df


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df['Genre'].value_counts()


# In[9]:


df['Director'].value_counts()


# In[10]:


df['Actors'].value_counts()


# In[11]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[12]:


df = df.fillna(df.median())


# # LOGISTIC REGRESSION

# In[13]:


df.columns


# In[14]:


x = df[['Year',
       'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)',
       'Metascore', 'Action', 'Adventure', 'Aniimation', 'Biography', 'Comedy',
       'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War',
       'Western']]
y = df['Success']


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,stratify=y)


# In[16]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)


# In[17]:


log.score(x_test,y_test)


# In[18]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,log.predict(x_test))


# In[19]:


sns.heatmap(clf,annot=True)


# # SOME OPTIMAZTIONS

# In[20]:


#normalising all columns
x_train_opt = x_train.copy()
x_test_opt = x_test.copy()


# In[21]:


from sklearn.preprocessing import StandardScaler
x_train_opt = StandardScaler().fit_transform(x_train_opt)
x_test_opt = StandardScaler().fit_transform(x_test_opt)


# In[22]:


#fitting again in Logistic Regression


# In[23]:


log.fit(x_train_opt,y_train)


# In[24]:


log.score(x_test_opt,y_test)


# Model Performance went down so we would not pursuit this more

# # KNN

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=40)
kn.fit(x_train,y_train)


# # DECISION TREE

# In[29]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree.score(x_test,y_test)


# In[30]:


tree.score(x_train,y_train)


# In[32]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,tree.predict(x_test))


# In[33]:


clf


# In[34]:


sns.heatmap(clf,annot=True)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("D:\\Black Friday Dataset\\train.csv")
df


# In[2]:


df.info()


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.head()


# In[8]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df['Gender'].unique()


# In[12]:


df['Gender']=df['Gender'].map({'M':1,'F':0})


# In[13]:


df


# In[14]:


df.info()


# In[15]:


df['Age'].unique()


# In[16]:


df['Age']=df['Age'].map({'0-17':0, '55+':1, '26-35':2, '46-50':3, '51-55':4, '36-45':5, '18-25':6})


# In[17]:


df


# In[18]:


df.info()


# In[19]:


df['City_Category'].unique()


# In[20]:


df_city=pd.get_dummies(df['City_Category'])
df_city


# In[21]:


df=pd.concat([df,df_city],axis=1)


# In[22]:


df


# In[23]:


df.drop(['City_Category'],axis=1,inplace=True)


# In[24]:


df


# In[25]:


df.info()


# In[26]:


df['Stay_In_Current_City_Years'].unique()


# In[27]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[28]:


df.info()


# In[29]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[30]:


df.info()


# In[31]:


l=['A','B','C']
for i in l:
    df[i]=df[i].astype(int)


# In[32]:


df.info()


# # graphs

# In[33]:


import seaborn as sns


# In[34]:


sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)


# In[35]:


sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# # null values

# In[67]:


df.isnull().sum()


# In[68]:


df['Product_Category_2'].mean().round()


# In[38]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mean().round(0))


# In[39]:


df.isnull().sum()


# In[40]:


df['Product_Category_3'].mean().round()


# In[41]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mean().round())


# In[42]:


df.isnull().sum()


# In[43]:


df.info()


# In[44]:


df['Product_ID']=df['Product_ID'].str.replace('P','')


# In[45]:


df.info()


# In[46]:


df


# In[47]:


df['Product_ID']=df['Product_ID'].astype(int)


# In[48]:


df.info()


# In[49]:


df


# In[50]:


df.head()


# # machine learning

# In[51]:


x=df.drop('Purchase',axis=1)
y=df['Purchase']


# In[52]:


y


# In[53]:


x


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[55]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[56]:


reg=RandomForestRegressor()
reg.fit(x_train,y_train)


# In[57]:


y_pred=reg.predict(x_test)
y_pred


# In[58]:


from sklearn.metrics import r2_score


# In[59]:


print(r2_score(y_pred,y_test))


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[61]:


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.1)


# In[62]:


reg=RandomForestRegressor()
reg.fit(x_train,y_train)


# In[ ]:





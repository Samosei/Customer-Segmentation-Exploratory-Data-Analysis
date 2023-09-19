#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[2]:


import matplotlib
print(matplotlib.get_backend())
import matplotlib
print(matplotlib.get_backend())
import matplotlib
matplotlib.use('nbagg')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('OVcustomers.csv')
df


# In[4]:


# Checking if there are any nulls

df.isnull().values.any()


# In[5]:


df1 = df[(df["Loyalty Member"] == 1)]
df0 = df[(df["Loyalty Member"] == 0)]


# In[6]:


# Checking for correlation between data columnns
df.corr()


# In[7]:


# Checking for correlation for loyalty group customers
df1.corr()


# In[8]:


# Checking for correlation for non-loyalty group customers
df0.corr()


# Only correlation is between av_on_web and  av_purch_price which is a strong positive relationship. The remaining data columns have no correlation

# In[9]:


# A seaborn pair plot
# https://seaborn.pydata.org/generated/seaborn.pairplot.html

import seaborn as sns
sns.pairplot(df)


# In[10]:


# A seaborn pair plot
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# loyalty group

import seaborn as sns
sns.pairplot(df1)


# In[11]:


# A seaborn pair plot
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# Non-loyalty group

import seaborn as sns
sns.pairplot(df0)


# In[12]:


# Creating Exploratory Data Analysis plot for the strong correlating values
# Grouping av_on_web

def group(av_on_web):
    if av_on_web <8:
        return "0-8"
    elif av_on_web > 8 and av_on_web <17:
        return "9-16"
    elif av_on_web > 16 and av_on_web <25:
        return "17-24"
    elif av_on_web > 24 and av_on_web <33:
        return "25-32"
    elif av_on_web > 32 and av_on_web <41:
        return "33-40"
    elif av_on_web > 40 and av_on_web <49:
        return "41-48"
    elif av_on_web > 48 and av_on_web <57:
        return "49-56"
    elif av_on_web > 56 and av_on_web <65:
        return "57-64"
    elif av_on_web > 64 and av_on_web <73:
        return "65-72"
    elif av_on_web > 72 and av_on_web <81:
        return "73-80"
    elif av_on_web > 80 and av_on_web <89:
        return "81-88"
    elif av_on_web > 89:
        return ">89"

df["av_on_web Group"] =df["av_on_web"].apply(group)
# To order plotly index
order = ["0-8", "9-16", "17-24", "25-32", "33-40", "41-48", "49-56", "57-64", "65-72", "73-80", "81-88",">89"]

mask = df.groupby("av_on_web Group")["av_purch_price"].mean()
mask = mask.reset_index()
fig = px.bar(data_frame=mask, x="av_on_web Group", y="av_purch_price", height=500)

annotation = []
for x, y in zip(mask["av_on_web Group"], mask["av_purch_price"]):
    annotation.append(
        dict(x=x, y=y + 30,
             text=str(round(y, 2)) + '$',
             font=dict(family='Arial', size=14, color='rgb(66, 99, 236)'), showarrow=False)
    )
fig.update_xaxes(categoryorder='array', categoryarray= order)
fig.update_layout(annotations=annotation)
fig.show()


# In[13]:


# K-clustering, checking for the optimal number of clusters to create customer groups. 3 is optimal from Elbow plot

data = df[["av_on_web", "av_purch_price"]]

df_log = np.log(data)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df_log)

errors = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_scaled)
    errors.append(model.inertia_)
    


plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(range(1, 11)), y=errors)
plt.savefig("Elbow.png")


# In[14]:


model = KMeans(n_clusters=3, random_state=42)
model.fit(df_scaled)


# In[15]:


data = data.assign(ClusterLabel= model.labels_)

data.groupby("ClusterLabel")[["av_on_web", "av_purch_price"]].median()


# In[16]:


# Plot displaying three customer segments 

fig = px.scatter(
    data_frame=data,
    x = "av_on_web",
    y= "av_purch_price",
    title = "Relationship between av_on_web VS av_purch_price",
    color = "ClusterLabel",
    height=500
)
fig.show()


# From av_on_web and av_purch_price attributes, there are three resulting customer segments with median values in table above.
# 
# Group 0 - Customers who spend medium time on web & medium average purchase price.
# 
# Group 1 - Customers who spend the highest time on web & high average purchase price.
# 
# Group 2 - Customers who spend the least time on web & spend thr least on purchase.
# 
# From this grouping, adverts with items with the highest prices should target customers with highest average time on web,
# medium priced items adverts should target customers with medium average time on web and items with the lowest prices should target customers with least average time on web

# In[17]:


# Creating Exploratory Data Analysis plot for av_purch_time and av_purch_price with no correlation value of 0.028317
# Grouping av_purch_time using a 3hr window within 24hrs of the day

def group(av_purch_time):
    if av_purch_time <3:
        return "0-2.98"
    elif av_purch_time > 2.98 and av_purch_time <6:
        return "3-5.98"
    elif av_purch_time > 5.98 and av_purch_time <9:
        return "6-8.98"
    elif av_purch_time > 8.98 and av_purch_time <12:
        return "9-11.98"
    elif av_purch_time > 11.98 and av_purch_time <15:
        return "12-14.98"
    elif av_purch_time > 14.98 and av_purch_time <18:
        return "15-17.98"
    elif av_purch_time > 17.98 and av_purch_time <21:
        return "18-20.98"
    elif av_purch_time > 20.98:
        return ">21"



df["av_purch_time Group"] =df["av_purch_time"].apply(group)

# To order plotly index
order = ["0-2.98","3-5.98", "6-8.98", "9-11.98", "12-14.98", "15-17.98", "18-20.98",">21"]

# Creating Exploratory Data Analysis plot for the strong correlating values

mask = df.groupby("av_purch_time Group")["av_purch_price"].median()
mask = mask.reset_index()
fig = px.bar(data_frame=mask, x="av_purch_time Group", y="av_purch_price", height=500)

annotation = []
for x, y in zip(mask["av_purch_time Group"], mask["av_purch_price"]):
    annotation.append(
        dict(x=x, y=y + 30,
             text=str(round(y, 2)) + '$',
             font=dict(family='Arial', size=14, color='rgb(66, 99, 236)'), showarrow=False)
    )
fig.update_xaxes(categoryorder='array', categoryarray= order)
fig.update_layout(annotations=annotation)
fig.show()


# From the plot above, within a 24hr day there is no distinct average purchase time group, there is no customer segment to create from this attribute comparison as shown in correlation.

# In[18]:


# K-clustering, checking for the optimal number of clusters to create customer groups.  
# 4 is optimal from Elbow plot for av_on_web & days_since_purch attribute comparison.

data = df[["av_on_web", "days_since_purch"]]

df_log = np.log(data)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df_log)

errors = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_scaled)
    errors.append(model.inertia_)
    


plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(range(1, 11)), y=errors)
plt.savefig("Elbow.png")


# In[19]:


model = KMeans(n_clusters=4, random_state=42)
model.fit(df_scaled)


# In[20]:


data = data.assign(ClusterLabel= model.labels_)
data.groupby("ClusterLabel")[["av_on_web", "days_since_purch"]].median()


# In[21]:


# Plot displaying four customer segments 

fig = px.scatter(
    data_frame=data,
    x = "av_on_web",
    y= "days_since_purch",
    title = "Relationship between av_on_web VS days_since_purch",
    color = "ClusterLabel",
    height=500
)
fig.show()


# From av_on_web and days_since_purch attributes, there are four resulting customer segments with median values in table above.
# 
# Group 0 - Customers who spend medium to the highest time on web & 122 days_since_purch(4 months)
# 
# Group 1 - Customers who spend the least time on web & 392 days_since_purch(13 months)
# 
# Group 2 - Customers who spend least to the highest time on web & 24 days_since_purch(less than a month)
# 
# Group 3 - Customers who spend medium to the highest time on web & 456 days_since_purch(16 months)
# 
# From these customer segments, customers who spend the highest time with an averagely high corresponding purchase price shows that it takes a 4 months to 16 months before the make another purchase therefore more adverts should target that group by marketting team.

# In[ ]:





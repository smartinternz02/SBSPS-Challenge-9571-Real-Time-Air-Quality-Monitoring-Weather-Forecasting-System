#!/usr/bin/env python
# coding: utf-8

# # Weather forecasting for the year 2018 using Decision Tree Regressor, Kmeans , Train and Test Methods.

# In[1]:


import numpy as np ## For Linear Algebra
import pandas as pd ## To Work With Data


# In[2]:


pip install plotly   
## For better visualization plotly has been used instead of Matplot library


# In[3]:


import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# In[5]:


df = pd.read_csv (r"F:\New Folder\Weather Data in India from 1901 to 2017.csv")


# In[12]:


df.head() ## Shows top 5 rows of the data set by default.


# In[6]:


## to get rid of the unnamed column do the following
df = pd.read_csv (r"D:\New folder\Weather Data in India from 1901 to 2017.csv",index_col=0)


# In[7]:


df.head()##This is how the dataset really looks


# In[8]:


## Creating an attribute that would contain date (month, year). So that we could get temprature values with the timeline
df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:]) ##melt() function is useful to message a DataFrame into a format where one or more columns are identifier variables


# In[9]:


df1.head()## New look of the dataset


# In[10]:


df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str) 
##A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression
df1.loc[:,'Date'] = df1['Date'].apply(lambda x : datetime.strptime(x, '%b %Y')) ## Converting String to datetime object
df1.head()


# # Now plotting temperature throughout the timeline

# In[11]:


df1.columns=['Year', 'Month', 'Temprature', 'Date']
df1.sort_values(by='Date', inplace=True) ## To get the time series right.
fig = go.Figure(layout = go.Layout(yaxis=dict(range=[0, df1['Temprature'].max()+1])))
fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temprature']), )
fig.update_layout(title='Temprature Throught Timeline:',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.update_layout(xaxis=go.layout.XAxis(
    rangeselector=dict(
        buttons=list([dict(label="Whole View", step="all"),
                      dict(count=1,label="One Year View",step="year",stepmode="todate")                      
                     ])),
        rangeslider=dict(visible=True),type="date")
)
fig.show()


# In[12]:


## it can be seen that the graph is distorted because this is how the plotted values look after plotting on the graph


# In[13]:


##We have four main seasons in India and this is how they are grouped:
##Winter : December, January and February.
##Summer(Also called, "Pre Monsoon Season") : March, April and May.
##Monsoon : June, July, August and September.
##Autumn(Also called "Post Monsoon Season) : October and November.


# In[14]:


fig = px.box(df1, 'Month', 'Temprature')
fig.update_layout(title='Warmest, Coldest and Median Monthly Tempratue.')
fig.show()


# In[15]:


from sklearn.cluster import KMeans
sse = []
target = df1['Temprature'].to_numpy().reshape(-1,1)
num_clusters = list(range(1, 10))

for k in num_clusters:
    km = KMeans(n_clusters=k)
    km.fit(target)
    sse.append(km.inertia_)

fig = go.Figure(data=[
    go.Scatter(x = num_clusters, y=sse, mode='lines'),
    go.Scatter(x = num_clusters, y=sse, mode='markers')
])

fig.update_layout(title="Evaluation on number of clusters:",
                 xaxis_title = "Number of Clusters:",
                 yaxis_title = "Sum of Squared Distance",
                 showlegend=False)
fig.show()


# In[16]:


km = KMeans(3)
km.fit(df1['Temprature'].to_numpy().reshape(-1,1))
df1.loc[:,'Temp Labels'] = km.labels_
fig = px.scatter(df1, 'Date', 'Temprature', color='Temp Labels')
fig.update_layout(title = "Temprature clusters.",
                 xaxis_title="Date", yaxis_title="Temprature")
fig.show()


# In[17]:


##Despite having 4 seasons we can see 3 main clusturs based on tempratures.
##Jan, Feb and Dec are the coldest months.
##Apr, May, Jun, Jul, Aug and Sep; all have hotter tempratures.
##Mar, Oct and Nov are the months that have tempratures neither too hot nor too cold


# In[18]:


fig = px.histogram(x=df1['Temprature'], nbins=200, histnorm='density')
fig.update_layout(title='Frequency chart of temprature readings:',
                 xaxis_title='Temprature', yaxis_title='Count')


# In[19]:


df['Yearly Mean'] = df.iloc[:,1:].mean(axis=1) ## Axis 1 for row wise and axis 0 for columns.
fig = go.Figure(data=[
    go.Scatter(name='Yearly Tempratures' , x=df['YEAR'], y=df['Yearly Mean'], mode='lines'),
    go.Scatter(name='Yearly Tempratures' , x=df['YEAR'], y=df['Yearly Mean'], mode='markers')
])
fig.update_layout(title='Yearly Mean Temprature :',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.show()

fig = px.scatter(df,x = 'YEAR', y = 'Yearly Mean', trendline = 'lowess', )
fig.update_layout(title='Trendline Over The Years :',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.show()


# # Seasonal Analysis

# In[20]:


fig = px.line(df1, 'Year', 'Temprature', facet_col='Month', facet_col_wrap=4)
fig.update_layout(title='Monthly temprature throught history:')
fig.show()


# In[21]:


df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
seasonal_df = df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']]
seasonal_df = pd.melt(seasonal_df, id_vars='YEAR', value_vars=seasonal_df.columns[1:])
seasonal_df.columns=['Year', 'Season', 'Temprature']


# In[22]:


fig = px.scatter(seasonal_df, 'Year', 'Temprature', facet_col='Season', facet_col_wrap=2, trendline='ols')
fig.update_layout(title='Seasonal mean tempratures throught years:')
fig.show()


# In[23]:


px.scatter(df1, 'Month', 'Temprature', size='Temprature', animation_frame='Year')


# # Forecasting weather for year 2018

# In[24]:


##  using decision tree regressor for prediction as the data does not actually have a linear trend.
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 

df2 = df1[['Year', 'Month', 'Temprature']].copy()
df2 = pd.get_dummies(df2)
y = df2[['Temprature']]
x = df2.drop(columns='Temprature')

dtr = DecisionTreeRegressor()
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.6)
dtr.fit(train_x, train_y)
pred = dtr.predict(test_x)
r2_score(test_y, pred)


# In[25]:


next_Year = df1[df1['Year']==2017][['Year', 'Month']]
next_Year.Year.replace(2017,2018, inplace=True)
next_Year= pd.get_dummies(next_Year)
temp_2018 = dtr.predict(next_Year)

temp_2018 = {'Month':df1['Month'].unique(), 'Temprature':temp_2018}
temp_2018=pd.DataFrame(temp_2018)
temp_2018['Year'] = 2018
temp_2018


# In[26]:


forecasted_temp = pd.concat([df1,temp_2018], sort=False).groupby(by='Year')['Temprature'].mean().reset_index()
fig = go.Figure(data=[
    go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp['Year'], y=forecasted_temp['Temprature'], mode='lines'),
    go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp ['Year'], y=forecasted_temp['Temprature'], mode='markers')
])
fig.update_layout(title='Forecasted Temprature:',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





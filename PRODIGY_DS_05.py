#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('C:/Users/Rajesh Gonnade/Downloads/only_road_accidents_data3.csv')
df


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


data=df.drop([ '0-3 hrs. (Night)', '3-6 hrs. (Night)',
       '6-9 hrs (Day)', '9-12 hrs (Day)', '12-15 hrs (Day)', '15-18 hrs (Day)',
       '18-21 hrs (Night)','21-24 hrs (Night)','YEAR'],axis=1)


# In[9]:


data.head()


# In[10]:


data=data.sort_values(by='Total',ascending=False)
data.head()


# In[16]:


plt.figure(figsize=(15,9))
sns.heatmap(df.corr(),annot=True)


# In[11]:


plt.figure(figsize=(20,10))

sns.barplot(x=data['STATE/UT'],y=data['Total'])
plt.xlabel('STATE/UT',size=20)
plt.ylabel('TOTAL',size=20)
plt.xticks(rotation=45)

plt.title('Cases of road accidents in each state/UT from 2001-14',size=20)
plt.show()


# In[12]:



# Group by 'STATE/UT' and sum the 'Total' accidents for each state/UT
accidents_by_state = df.groupby('STATE/UT')['Total'].sum()

# Find the state/UT with the highest number of accidents
state_with_highest_accidents = accidents_by_state.idxmax()
highest_accidents_count = accidents_by_state.max()

print(f"The state/UT with the highest number of accidents is {state_with_highest_accidents} with {highest_accidents_count} accidents.")


# In[13]:



# Group by 'STATE/UT' and sum the 'Total' accidents for each state/UT
accidents_by_state = df.groupby('STATE/UT')['Total'].sum().reset_index()

# Sort the dataframe by the total number of accidents
accidents_by_state = accidents_by_state.sort_values(by='Total', ascending=False)

# Plot the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Total', y='STATE/UT', data=accidents_by_state, palette='viridis')
plt.xlabel('Number of Accidents')
plt.ylabel('State/UT')
plt.title('Total Number of Accidents by State/UT')
plt.show()


# In[14]:


# LINEAR REGRESSION MODEL

from scipy import stats
df=pd.read_csv('C:/Users/Rajesh Gonnade/Downloads/only_road_accidents_data3.csv')
x=df['YEAR'].values
y=df['Total'].values
slope,intercept,r,p,std_err = stats.linregress(x,y)
def myfunc(x):
    return slope*x + intercept
mymodel=list(map(myfunc,x))
plt.scatter(x,y)
plt.plot(x,mymodel,color='red')
plt.xlabel('Year')
plt.ylabel('Total Accidents')
plt.title('Linear Regression: Year vs Total Accidents')
plt.show()

print("Slope:", slope)
print("Intercept:", intercept)
print("r:", r)


# In[15]:


import pandas as pd
from scipy import stats

# Load your data
df = pd.read_csv('C:/Users/Rajesh Gonnade/Downloads/only_road_accidents_data3.csv')

# Extract 'YEAR' and 'Total' columns
x = df['YEAR'].values
y = df['Total'].values

# Train the linear regression model
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Define a function for making predictions
def predict_total(year):
    return slope * year + intercept

# Make predictions for new 'YEAR' values
new_years = [2022, 2023, 2024]
predicted_totals = [predict_total(year) for year in new_years]

# Display the predictions
for year, total in zip(new_years, predicted_totals):
    print(f"Year: {year}, Predicted Total: {total}")


# In[ ]:





# In[ ]:





# In[ ]:





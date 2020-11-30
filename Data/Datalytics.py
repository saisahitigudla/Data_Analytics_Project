#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import datetime
import json
import re
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#hiding warnings for clean display
warnings.filterwarnings('ignore')
#to configure some options
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
#for interactive plots %matplotlib notebook

#hiding warnings for clean display
warnings.filterwarnings('ignore')
df_10=pd.read_csv('Desktop/da/2010_car_data.csv')
df_20=pd.read_csv('Desktop/da/2020_car_data.csv')


# In[2]:


#Shape of 2010 dataset and 2020 dataset:
df_10.shape,df_20.shape


# In[3]:


#2010 Dataset:
df_10.head()


# In[4]:


#2020 Dataset
df_20.head()


# In[5]:


#duplicates of 2010 dataset
#we find the duplicated value count in 2010 dataset
df_10.duplicated().sum()


# In[6]:


#null value count of 2010 dataset
df_10.isnull().sum()


# In[7]:


#data types of 2010 dataset
df_10.dtypes


# In[8]:


#check unique numbers for each column
df_10.nunique()


# In[9]:


#duplicates of 2020 dataset
df_20.duplicated().sum()


# In[10]:


#null value of 2020 dataset
df_20.isnull().sum()


# In[11]:


#data types of 2020 dataset
df_20.dtypes


# In[12]:


#check unique numbers for each column
df_20.nunique()


# In[13]:


#view the datasets
#2010 dataset
df_10.head(1).columns


# In[14]:


#2020 dataset
df_20.head(1).columns


# In[15]:


df_10['Averaging Group ID'].isna().sum(),df_20['Averaging Group ID'].isna().sum()


# In[16]:


df_10['Averaging Weighting Factor'].isna().sum(),df_20['Averaging Weighting Factor'].isna().sum()


# In[3]:


df_10.describe()


# In[4]:


df_20.describe()


# In[17]:


#1. dropping extraneous columns from 2020 dataset
df_20.drop(['DT-Inertia Work Ratio Rating','DT-Absolute Speed Change Ratg', 'DT-Energy Economy Rating','Averaging Group ID','Averaging Weighting Factor',
'ADFE Test Number','ADFE Total Road Load HP','ADFE Equiv. Test Weight (lbs.)','ADFE N/V Ratio','PM (g/mi)','CH4 (g/mi)','N2O (g/mi)','FE Bag 4'],axis=1,inplace=True)
df_10.drop(['Averaging Group ID','Averaging Weighting Factor','ADFE Test Number','ADFE Total Road Load HP', 'ADFE Equiv. Test Weight (lbs.)','ADFE N/V Ratio','PM (g/mi)','CH4 (g/mi)','N2O (g/mi)','FE Bag 4'],axis=1,inplace=True)
df_10.shape,df_20.shape


# In[18]:


#replace spaces with underscores and lowercase labels for 2010 dataset
df_10.rename(columns=lambda x: x.strip().lower().replace(" ","_"),inplace=True)
df_10.head(1)


# In[19]:


#replace spaces with underscores and lowercase labels for 2020 dataset
df_20.rename(columns=lambda x: x.strip().lower().replace(" ","_"),inplace=True)
df_20.head(1)


# In[20]:


#confirm if column labels for 2010 and 2020 datasets are identical or not
df_10.columns==df_20.columns


# In[21]:


#making sure if all are identical
(df_10.columns==df_20.columns).all()
df_10


# In[22]:


#creating new datasets for further section

df_10.to_csv('Desktop/da/car_data_2010_v1.csv',index=False)
df_20.to_csv('Desktop/da/car_data_2020_v2.csv',index=False)
df_10.shape,df_20.shape


# In[23]:


#load datasets
import pandas as pd
df_10_v1=pd.read_csv('Desktop/da/car_data_2010_v1.csv')
df_20_v1=pd.read_csv('Desktop/da/car_data_2020_v2.csv')
df_10_v1.shape,df_20_v1.shape


# In[24]:


#count of null values in columns of both datasets
df_10_v1.isna().sum(), df_20_v1.isna().sum()


# In[25]:


#drop rows with any null values in both datasets
#fill the NaN values with with respective most frequent values
df_10_v1['#_of_cylinders_and_rotors']=df_10_v1['#_of_cylinders_and_rotors'].fillna(df_10_v1['#_of_cylinders_and_rotors'].mode()[0])
df_20_v1['#_of_cylinders_and_rotors']=df_20_v1['#_of_cylinders_and_rotors'].fillna(df_20_v1['#_of_cylinders_and_rotors'].mode()[0])
df_10_v1['thc_(g/mi)']=df_10_v1['thc_(g/mi)'].fillna(df_10_v1['thc_(g/mi)'].mode()[0])
df_20_v1['thc_(g/mi)']=df_20_v1['thc_(g/mi)'].fillna(df_20_v1['thc_(g/mi)'].mode()[0])
df_10_v1['co_(g/mi)']=df_10_v1['co_(g/mi)'].fillna(df_10_v1['co_(g/mi)'].mode()[0])
df_20_v1['co_(g/mi)']=df_20_v1['co_(g/mi)'].fillna(df_20_v1['co_(g/mi)'].mode()[0])
df_10_v1['co2_(g/mi)']=df_10_v1['co2_(g/mi)'].fillna(df_10_v1['co2_(g/mi)'].mode()[0])
df_20_v1['co2_(g/mi)']=df_20_v1['co2_(g/mi)'].fillna(df_20_v1['co2_(g/mi)'].mode()[0])
df_10_v1['nox_(g/mi)']=df_10_v1['nox_(g/mi)'].fillna(df_10_v1['nox_(g/mi)'].mode()[0])
df_20_v1['nox_(g/mi)']=df_20_v1['nox_(g/mi)'].fillna(df_20_v1['nox_(g/mi)'].mode()[0])
df_10_v1['fe_bag_1']=df_10_v1['fe_bag_1'].fillna(df_10_v1['fe_bag_1'].mode()[0])
df_20_v1['fe_bag_1']=df_20_v1['fe_bag_1'].fillna(df_20_v1['fe_bag_1'].mode()[0])
df_10_v1['fe_bag_2']=df_10_v1['fe_bag_2'].fillna(df_10_v1['fe_bag_2'].mode()[0])
df_20_v1['fe_bag_2']=df_20_v1['fe_bag_2'].fillna(df_20_v1['fe_bag_2'].mode()[0])
df_10_v1['fe_bag_3']=df_10_v1['fe_bag_3'].fillna(df_10_v1['fe_bag_3'].mode()[0])
df_20_v1['fe_bag_3']=df_20_v1['fe_bag_3'].fillna(df_20_v1['fe_bag_3'].mode()[0])
df_10_v1['aftertreatment_device_cd']=df_10_v1['aftertreatment_device_cd'].fillna(df_10_v1['aftertreatment_device_cd'].mode()[0])
df_20_v1['aftertreatment_device_cd']=df_20_v1['aftertreatment_device_cd'].fillna(df_20_v1['aftertreatment_device_cd'].mode()[0])
df_10_v1['aftertreatment_device_desc']=df_10_v1['aftertreatment_device_desc'].fillna(df_10_v1['aftertreatment_device_desc'].mode()[0])
df_20_v1['aftertreatment_device_desc']=df_20_v1['aftertreatment_device_desc'].fillna(df_20_v1['aftertreatment_device_desc'].mode()[0])


# In[26]:


#check for any more null values
df_10_v1.isna().sum(), df_20_v1.isna().sum()


# In[27]:


#shape of datasets after filling NaN values
df_10_v1.shape,df_20_v1.shape


# In[28]:


#dropping rows with null values in 2010 dataset
df_10_v1.dropna(inplace=True)
#dropping rows with null values in 2020 dataset
df_20_v1.dropna(inplace=True)


# In[29]:


#shape of datasets after dropping null-valued rows
df_10_v1.shape,df_20_v1.shape


# In[30]:


#check for null values in any columns of both datasets
df_10_v1.isnull().sum().any(), df_20_v1.isnull().sum().any()


# In[31]:


#dedupe data :drop duplicates
df_10_v1.drop_duplicates(inplace=True)
df_20_v1.drop_duplicates(inplace=True)


# In[32]:


#checking for any duplicated values in datasets
df_10_v1.duplicated().sum(),df_20_v1.duplicated().sum()


# In[33]:


#shape of both datasets
df_10_v1.shape,df_20_v1.shape


# In[34]:


#creating new datasets for further section
df_10_v1.to_csv('Desktop/da/car_data_2010_v3.csv',index=False)
df_20_v1.to_csv('Desktop/da/car_data_2020_v4.csv',index=False)


# In[35]:


#load datasets
import pandas as pd
df_10_v2=pd.read_csv('Desktop/da/car_data_2010_v3.csv')
df_20_v2=pd.read_csv('Desktop/da/car_data_2020_v4.csv')


# In[36]:


#2010 dataset old
df_10_v1.head()


# In[37]:


#2020 dataset old
df_20_v1.head()


# In[38]:


#2010 dataset revised
df_10_v2.head()


# In[39]:


#2020 dataset revised
df_20_v2.head()


# In[40]:


#shape of revised datasets
df_10_v2.shape,df_20_v2.shape


# In[41]:


#locate a random row in 2010 dataset
df_10_v2.iloc[400]


# In[42]:


# Save your final CLEAN datasets as new files!
df_10_v2.to_csv('Desktop/da/clean_10.csv', index=False)
df_20_v2.to_csv('Desktop/da/clean_20.csv', index=False)


# In[43]:


#read datasets for further visualization
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')


# In[44]:


#visualization
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[45]:


#all possible histograms of 2010 dataset
cdf_10.hist(figsize=(20,20))


# In[46]:


#all possible histograms of 2020 dataset
cdf_20.hist(figsize=(20,20))


# In[47]:


plt.scatter(data = cdf_10 , x='axle_ratio' , y = 'test_veh_displacement_(l)') ;
plt.title('The relationship between Axle ratio and Test Vehicle Displacement in 2010')


# In[48]:


plt.scatter(data = cdf_20 , x='axle_ratio' , y = 'test_veh_displacement_(l)') ;
plt.title('The relationship between Axle ratio and Test Vehicle Displacement in 2020')


# In[49]:


plt.scatter( data = cdf_10 , x='co2_(g/mi)' , y = 'fe_bag_1') ;
plt.title('fe bag1 mpg against co2 score in 2010')


# In[50]:


plt.scatter( data = cdf_20 , x='co2_(g/mi)' , y = 'fe_bag_1') ;
plt.title('fe bag1 mpg against co2 score in 2020')


# In[48]:


import seaborn as sns
from warnings import filterwarnings
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
#read datasets for further visualization
cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')
sns.set_style('darkgrid')

import pylab as plt
import seaborn as sns

#default estimator mean
#plt.bar(cdf_10['vehicle_manufacturer_name'],cdf_10['co2_(g/mi)'],10)
sns.factorplot(x='co2_(g/mi)',y='vehicle_manufacturer_name',size=6,aspect=2,kind='bar',data=cdf_10,palette='plasma')
#bar plot of vehicle manufacturer name against co2 emission in 2010 dataset
#standard deviation
#sns.barplot(x='represented_test_veh_model',y='co2_(g/mi)',data=cdf_10,palette='plasma',estimator=np.std)


# In[52]:


#bar plot of vehicle manufacturer name against co in 2010 dataset
sns.factorplot(x='co_(g/mi)',y='vehicle_manufacturer_name',size=6,aspect=2,kind='bar',data=cdf_10,palette='plasma')


# In[53]:


#bar plot of vehicle manufacturer name against co2 emission in 2020 dataset
sns.factorplot(x='co2_(g/mi)',y='vehicle_manufacturer_name',size=6,aspect=2,kind='bar',data=cdf_20,palette='plasma')


# In[49]:


#bar plot of vehicle manufacturer name against co emission in 2020 dataset
sns.factorplot(x='co_(g/mi)',y='vehicle_manufacturer_name',size=6,aspect=2,kind='bar',data=cdf_20,palette='plasma')


# In[50]:


#bar plot of number of cylinders and rotors against analytically derived fe for 2010 dataset
g=sns.catplot(data=cdf_10,kind='bar',x="analytically_derived_fe?",y="#_of_cylinders_and_rotors",hue="analytically_derived_fe?",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("analytically_derived_fe?","#_of_cylinders_and_rotors")
#g.legend.set_title("")


# In[51]:


#bar plot of number of cylinders and rotors against analytically derived fe for 2020 dataset
g=sns.catplot(data=cdf_20,kind='bar',x="analytically_derived_fe?",y="#_of_cylinders_and_rotors",hue="analytically_derived_fe?",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("analytically_derived_fe?","#_of_cylinders_and_rotors")
#g.legend.set_title("")


# In[52]:


#violin plot of number of cylinders and rotors against vehicle type for 2010 dataset
g=sns.factorplot(data=cdf_10,kind='violin',x="vehicle_type",y="#_of_cylinders_and_rotors",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","#_of_cylinders_and_rotors")
#g.legend.set_title("vehicles with number of rotors and cylinders")
print("vehicles with number of rotors and cylinders 2010")


# In[53]:


#violin plot of number of cylinders and rotors against vehicle type for 2020 dataset
g=sns.factorplot(data=cdf_20,kind='violin',x="vehicle_type",y="#_of_cylinders_and_rotors",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","#_of_cylinders_and_rotors")
#g.legend.set_title("vehicles with number of rotors and cylinders")
print("vehicles with number of rotors and cylinders 2020")


# In[54]:


#bar plot of co2 production against vehicle type for 2010 dataset
g=sns.factorplot(data=cdf_10,kind='bar',x="vehicle_type",y="co2_(g/mi)",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","co2_(g/mi)")
#g.legend.set_title("vehicles with amount of co2_(g/mi)")
print("vehicles with amount of co2_(g/mi) 2010")


# In[55]:


#bar plot of co2 production against vehicle type for 2020 dataset
g=sns.factorplot(data=cdf_20,kind='bar',x="vehicle_type",y="co2_(g/mi)",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","co2_(g/mi)")
#g.legend.set_title("vehicles with amount of co2_(g/mi)")
print("vehicles with amount of co2_(g/mi) 2020")


# In[56]:


#bar plot of co production against vehicle type for 2010 dataset
g=sns.factorplot(data=cdf_10,kind='bar',x="vehicle_type",y="co_(g/mi)",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","co_(g/mi)")
#g.legend.set_title("vehicles with amount of co_(g/mi)")
print("vehicles with amount of co_(g/mi) 2010")


# In[57]:


#bar plot of co production against vehicle type for 2020 dataset
g=sns.factorplot(data=cdf_20,kind='bar',x="vehicle_type",y="co_(g/mi)",hue="vehicle_type",ci="sd",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","co_(g/mi)")
#g.legend.set_title("vehicles with amount of co_(g/mi)")
print("vehicles with amount of co_(g/mi) 2020")


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import datetime
import json
import re
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#hiding warnings for clean display
warnings.filterwarnings('ignore')

cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')
#def reshaped(excel_obj,i):
#to drop column video removed or error
#sns.countplot(cdf_10['Result'],label="Count")

corr = cdf_10.iloc[:,2:].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
            cmap = colormap, linewidths=0.2, linecolor='white')
plt.title("Correlation of 2010 models' features", y=1.05, size=15)  

corr = cdf_20.iloc[:,2:].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
            cmap = colormap, linewidths=0.2, linecolor='white')
plt.title("Correlation of 2020 models' features", y=1.05, size=15)  


# In[58]:


plt.scatter( data = cdf_10 , x='co2_(g/mi)' , y = 'rated_horsepower') ;
#plt.title('rated horsepower against co2 score in 2010')
plt.scatter( data = cdf_20 , x='co2_(g/mi)' , y = 'rated_horsepower') ;
plt.title('rated horsepower against co2 score')


# In[59]:


plt.scatter( data = cdf_10 , x='co2_(g/mi)' , y = 'test_veh_displacement_(l)') ;
#plt.title('rated horsepower against co2 score in 2010')
plt.scatter( data = cdf_20 , x='co2_(g/mi)' , y = 'test_veh_displacement_(l)') ;
plt.title('vehicle displacement against co2 score')


# In[60]:


plt.scatter( data = cdf_10 , x='test_veh_displacement_(l)' , y = 'rated_horsepower') ;
plt.title('rated horsepower against test_veh_displacement_(l) in 2010')


# In[61]:


plt.scatter( data = cdf_20 , x='test_veh_displacement_(l)' , y = 'rated_horsepower') ;
plt.title('rated horsepower against test_veh_displacement_(l) 2020')


# In[1]:


#prediction algorithm
import pandas as pd
cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')
#cleaned 2010 dataset
cdf_10


# In[20]:


#setting up dependent and independent variables
#X=pd.DataFrame(cdf_10['test_veh_displacement_(l)'],cdf_10['target_coef_a_(lbf)'],cdf_10['target_coef_b_(lbf/mph)'],cdf_10['target_coef_c_(lbf/mph**2)'],cdf_10['set_coef_a_(lbf)'],cdf_10['set_coef_b_(lbf/mph)'],cdf_10['set_coef_c_(lbf/mph**2)'],cdf_10['rated_horsepower'],cdf_10['#_of_cylinders_and_rotors'],cdf_10['#_of_gears'],cdf_10['axle_ratio'],cdf_10['n/v_ratio'],cdf_10['equivalent_test_weight_(lbs.)'])
#y=pd.DataFrame(cdf_10['fe_bag_1'])


# In[40]:


#summary statistics of 2010 dataset
cdf_10.describe()


# In[2]:


cdf_20.describe()


# In[41]:


#count of vehicles of each type
pd.value_counts(cdf_10['vehicle_type']).plot.bar()


# In[42]:


#count of number of cylinders and rotors
pd.value_counts(cdf_10['#_of_cylinders_and_rotors']).plot.bar()


# In[43]:


import seaborn as sns
#boxplot of vehicle type against number of cylinders and rotors of 2010 dataset
sns.boxplot(x='vehicle_type',y='#_of_cylinders_and_rotors',data=cdf_10)


# In[3]:


data=cdf_10


# In[4]:


data.drop(['model_year','vehicle_manufacturer_name','veh_mfr_code','represented_test_veh_make','represented_test_veh_model','test_vehicle_id','actual_tested_testgroup','engine_code','tested_transmission_type_code','tested_transmission_type','transmission_lockup?','drive_system_code','drive_system_description','transmission_overdrive_desc','shift_indicator_light_use_desc','test_number','test_originator','analytically_derived_fe?','test_procedure_description','test_fuel_type_description','test_category','aftertreatment_device_cd','aftertreatment_device_desc','police_-_emergency_vehicle?','averaging_method_cd'],axis=1,inplace=True)


# In[5]:


#prepare the 2010 dataset for modelling
data


# In[6]:


#count of type of vehicles 
data['vehicle_type'].value_counts()


# In[7]:


#count of averaging method descending
data['averging_method_desc'].value_counts()


# In[8]:


#mpg count
data['fe_unit'].value_counts()


# In[9]:


#drop fe_unit column 
data.drop(['fe_unit'],axis=1,inplace=True)


# In[10]:


#updates list of columns
data.columns


# In[11]:


#replacing the categorical values with integer substitutes
cleanup_data={"vehicle_type":{"Car":1,"Truck":2,"Both":3},"averging_method_desc":{"No averaging":0,"Harmonic averaging (1/(Sum(i=1 to n) (FET(i) / WT(i)))":1,"Simple averaging (Sum(i=1 to n) (FET(i)  *  WT(i))) ":2}}


# In[12]:


#replacing the categorical values with integer substitutes
data.replace(cleanup_data,inplace=True)
data.head()


# In[13]:


#updates datatypes in 2010 dataset
data.dtypes


# In[14]:


#importing packages necessary for model building
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[59]:


#develop a test-train split as 70-30 split
from sklearn.model_selection import train_test_split
training,test=train_test_split(data,train_size=0.7,test_size=0.3,shuffle=True)
training,valid=train_test_split(training,train_size=0.7,test_size=0.3,shuffle=True)
training_label=training.pop('fe_bag_1')
test_label=test.pop('fe_bag_1')
valid_label=valid.pop('fe_bag_1')


# In[61]:


#testing for performance of different models
#decision trees
dtc=DecisionTreeRegressor()
#random forest regression
rfc=RandomForestRegressor()
#k-nearest neighbors
knn=KNeighborsRegressor()
#linear regression
lr=LinearRegression()
#fitting the models
dtc.fit(training,training_label)
rfc.fit(training,training_label)
knn.fit(training,training_label)
lr.fit(training,training_label)


# In[62]:


#predictions
dtc_predict=dtc.predict(test)
rfc_predict=rfc.predict(test)
knn_predict=knn.predict(test)
lr_predict=lr.predict(test)


# In[63]:


from sklearn.metrics import mean_squared_error
import math
accuracy=dict()
#finding accuarcy using mean squared error for each of the models of test sets
accuracy['Decision Tree']=math.sqrt(mean_squared_error(test_label,dtc_predict))
accuracy['Random Forest']=math.sqrt(mean_squared_error(test_label,rfc_predict))
accuracy['K nearest neigbor']=math.sqrt(mean_squared_error(test_label,knn_predict))
accuracy['Linear Regression']=math.sqrt(mean_squared_error(test_label,lr_predict))
print(accuracy)


# In[68]:


#validation testing
dtc_predict=dtc.predict(valid)
rfc_predict=rfc.predict(valid)
knn_predict=knn.predict(valid)
lr_predict=lr.predict(valid)
#accuracy1
accuracy1=dict()
#finding accuarcy using mean squared error for each of the models of validation sets
accuracy1['Decision Tree']=math.sqrt(mean_squared_error(valid_label,dtc_predict))
accuracy1['Random Forest']=math.sqrt(mean_squared_error(valid_label,rfc_predict))
accuracy1['K nearest neigbor']=math.sqrt(mean_squared_error(valid_label,knn_predict))
accuracy1['Linear Regression']=math.sqrt(mean_squared_error(valid_label,lr_predict))
print(accuracy1)


# In[69]:


#result of prediction using decision trees
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':dtc.predict(valid)})
results.head()


# In[70]:


#results of prediction using random forest method
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':rfc.predict(valid)})
results.head()


# In[71]:


#results of prediction using k-nearest neighbor
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':knn.predict(valid)})
results.head()


# In[72]:


#result of prediction using linear regression
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':lr.predict(valid)})
results.head()


# In[75]:


#performance plot of all algorithms
from matplotlib import pyplot as plt
import seaborn as sns
fig,(ax1)=plt.subplots(ncols=1,sharey=True,figsize=(15,5))
new_data=pd.DataFrame(list(accuracy1.items()),columns=['Algorithms','Percentage'])
display(new_data)
sns.barplot(x='Algorithms',y='Percentage',data=new_data,ax=ax1);


# In[77]:


#accurate algorithm 
max_accuracy=min(accuracy1,key=accuracy1.get)
max_accuracy


# In[79]:


#dataset
data.head()


# In[80]:


#implementing the random forest regression model
dataset=data
X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values


# In[81]:


#generating test-train split as 80-20 split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[82]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[88]:


#prediction using random forest regression and the results
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
outputs=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
outputs.head()


# In[89]:


#finding accuracies using mean absolute error, mean squared error and root mean squared error
from sklearn import metrics
import numpy as np
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[98]:


#testing the same random forest algorithm with more estimators and prediction results
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
outputs=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
outputs.head()


# In[99]:


#finding accuracies using mean absolute error, mean squared error and root mean squared error for updated estimators count
from sklearn import metrics
import numpy as np
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[25]:


#multiple linear regression for the same dataset
lin_dataset=data
lin_dataset.drop(['averging_method_desc'],axis=1,inplace=True)
X=pd.DataFrame(lin_dataset.iloc[:,:-1])
y=pd.DataFrame(lin_dataset.iloc[:,-1])


# In[26]:


#explanatory variables
X


# In[27]:


#outcome variables
y


# In[28]:


#develop test train split as 80-20 split with multiple linear regression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

v=pd.DataFrame(regressor.coef_,index=['Co-efficient']).transpose()
w=pd.DataFrame(X.columns,columns=['Attribute'])


# In[29]:


#finding the coefficient values
coeff_df=pd.concat([w,v],axis=1,join='inner')
coeff_df


# In[30]:


#print(regressor.intercept_)

#print(regressor.coef_)
#estimate the predicted values
y_pred=regressor.predict(X_test)
y_pred=pd.DataFrame(y_pred,columns=['predicted'])
y_pred


# In[31]:


#test values
y_test


# In[32]:


#regression intercept and regression coefficient values
print(regressor.intercept_)

print(regressor.coef_)


# In[33]:


#finding accuracies using mean absolute error, mean squared error and root mean squared error for multiple linear regression model
from sklearn import metrics
import numpy as np
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[42]:


#consider 2020 dataset
data20=cdf_20


# In[43]:


data20


# In[44]:


#testing performance of differnt algorithms and getting the best accurate algorithm
data20=cdf_20
data20.drop(['model_year','vehicle_manufacturer_name','veh_mfr_code','represented_test_veh_make','represented_test_veh_model','test_vehicle_id','actual_tested_testgroup','engine_code','tested_transmission_type_code','tested_transmission_type','transmission_lockup?','drive_system_code','drive_system_description','transmission_overdrive_desc','shift_indicator_light_use_desc','test_number','test_originator','analytically_derived_fe?','test_procedure_description','test_fuel_type_description','test_category','aftertreatment_device_cd','aftertreatment_device_desc','police_-_emergency_vehicle?','averaging_method_cd'],axis=1,inplace=True)
data20
data20['vehicle_type'].value_counts()
data20['averging_method_desc'].value_counts()
data20['fe_unit'].value_counts()
data20.drop(['fe_unit'],axis=1,inplace=True)
data20.columns
cleanup_data20={"vehicle_type":{"Car":1,"Truck":2,"Both":3},"averging_method_desc":{"No averaging":0,"Harmonic averaging (1/(Sum(i=1 to n) (FET(i) / WT(i)))":1,"Simple averaging (Sum(i=1 to n) (FET(i)  *  WT(i))) ":2}}
data20.replace(cleanup_data,inplace=True)
data20.head()
data20.dtypes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
training,test=train_test_split(data20,train_size=0.7,test_size=0.3,shuffle=True)
training,valid=train_test_split(training,train_size=0.7,test_size=0.3,shuffle=True)
training_label=training.pop('fe_bag_1')
test_label=test.pop('fe_bag_1')
valid_label=valid.pop('fe_bag_1')
dtc=DecisionTreeRegressor()
rfc=RandomForestRegressor()
knn=KNeighborsRegressor()
lr=LinearRegression()
dtc.fit(training,training_label)
rfc.fit(training,training_label)
knn.fit(training,training_label)
lr.fit(training,training_label)
dtc_predict=dtc.predict(test)
rfc_predict=rfc.predict(test)
knn_predict=knn.predict(test)
lr_predict=lr.predict(test)
from sklearn.metrics import mean_squared_error
import math
accuracy=dict()
accuracy['Decision Tree']=math.sqrt(mean_squared_error(test_label,dtc_predict))
accuracy['Random Forest']=math.sqrt(mean_squared_error(test_label,rfc_predict))
accuracy['K nearest neigbor']=math.sqrt(mean_squared_error(test_label,knn_predict))
accuracy['Linear Regression']=math.sqrt(mean_squared_error(test_label,lr_predict))
print(accuracy)
#validation testing
dtc_predict=dtc.predict(valid)
rfc_predict=rfc.predict(valid)
knn_predict=knn.predict(valid)
lr_predict=lr.predict(valid)
#accuracy1
accuracy1=dict()
accuracy1['Decision Tree']=math.sqrt(mean_squared_error(valid_label,dtc_predict))
accuracy1['Random Forest']=math.sqrt(mean_squared_error(valid_label,rfc_predict))
accuracy1['K nearest neigbor']=math.sqrt(mean_squared_error(valid_label,knn_predict))
accuracy1['Linear Regression']=math.sqrt(mean_squared_error(valid_label,lr_predict))
print(accuracy1)
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':dtc.predict(valid)})
results.head()
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':rfc.predict(valid)})
results.head()
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':knn.predict(valid)})
results.head()
results=pd.DataFrame({'label fe_bag_1 mpg':valid_label,'prediction value':lr.predict(valid)})
results.head()
#performance plot of algorithms
from matplotlib import pyplot as plt
import seaborn as sns
fig,(ax1)=plt.subplots(ncols=1,sharey=True,figsize=(15,5))
new_data=pd.DataFrame(list(accuracy1.items()),columns=['Algorithms','Percentage'])
display(new_data)
sns.barplot(x='Algorithms',y='Percentage',data=new_data,ax=ax1);
max_accuracy=min(accuracy1,key=accuracy1.get)
max_accuracy


# In[46]:


#performing random forest regression
dataset_20=data
X=dataset_20.iloc[:,0:4].values
y=dataset_20.iloc[:,4].values
#generate test-train split as 80-20 split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
outputs=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
#prediction and results along with mean absolute error, MSE and RMSE values and knowing the accuracy
from sklearn import metrics
import numpy as np
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
outputs.head()


# In[47]:


#finding accuracies using mean absolute error, mean squared error and root mean squared error for updated estimators count
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
outputs=pd.DataFrame({'y_pred':y_pred,'y_test':y_test})
from sklearn import metrics
import numpy as np
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
outputs.head()


# In[35]:



#correlation of features with updated dataset features
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import datetime
import json
import re
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#hiding warnings for clean display
warnings.filterwarnings('ignore')

cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')
#def reshaped(excel_obj,i):
#to drop column video removed or error
#sns.countplot(cdf_10['Result'],label="Count")
data1=cdf_10
data1.drop(['model_year','vehicle_manufacturer_name','veh_mfr_code','represented_test_veh_make','represented_test_veh_model','test_vehicle_id','actual_tested_testgroup','engine_code','tested_transmission_type_code','tested_transmission_type','transmission_lockup?','drive_system_code','drive_system_description','transmission_overdrive_desc','shift_indicator_light_use_desc','test_number','test_originator','analytically_derived_fe?','test_procedure_description','test_fuel_type_description','test_category','aftertreatment_device_cd','aftertreatment_device_desc','police_-_emergency_vehicle?','averaging_method_cd'],axis=1,inplace=True)
corr = data1.iloc[:,2:].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
            cmap = colormap, linewidths=0.2, linecolor='white')
plt.title("Correlation of 2010 models' features", y=1.05, size=15)  
data2=cdf_20
data2.drop(['model_year','vehicle_manufacturer_name','veh_mfr_code','represented_test_veh_make','represented_test_veh_model','test_vehicle_id','actual_tested_testgroup','engine_code','tested_transmission_type_code','tested_transmission_type','transmission_lockup?','drive_system_code','drive_system_description','transmission_overdrive_desc','shift_indicator_light_use_desc','test_number','test_originator','analytically_derived_fe?','test_procedure_description','test_fuel_type_description','test_category','aftertreatment_device_cd','aftertreatment_device_desc','police_-_emergency_vehicle?','averaging_method_cd'],axis=1,inplace=True)
corr = data2.iloc[:,2:].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
            cmap = colormap, linewidths=0.2, linecolor='white')
plt.title("Correlation of 2020 models' features", y=1.05, size=15)  


# In[2]:


#2010
plt.scatter( data = data1 , x='co2_(g/mi)' , y = 'rnd_adj_fe') ;
plt.title('rnd_adj_fe vs. co2 production')


# In[3]:


#2020
plt.scatter( data = data2 , x='co2_(g/mi)' , y = 'rnd_adj_fe') ;
plt.title('rnd_adj_fe vs. co2 production')


# In[4]:


#2010
plt.scatter( data = data1 , x='rated_horsepower' , y = '#_of_cylinders_and_rotors') ;
plt.title('rated horsepower vs. number of cylinders and rotors')


# In[5]:


#2020
plt.scatter( data = data2 , x='rated_horsepower' , y = '#_of_cylinders_and_rotors') ;
plt.title('rated horsepower vs. number of cylinders and rotors')


# In[1]:


#modifying the 2020 dataset 
import pandas as pd
cdf_10=pd.read_csv('Desktop/da/clean_10.csv')
cdf_20=pd.read_csv('Desktop/da/clean_20.csv')
data20=cdf_20
data20.drop(['model_year','vehicle_manufacturer_name','veh_mfr_code','represented_test_veh_make','represented_test_veh_model','test_vehicle_id','actual_tested_testgroup','engine_code','tested_transmission_type_code','tested_transmission_type','transmission_lockup?','drive_system_code','drive_system_description','transmission_overdrive_desc','shift_indicator_light_use_desc','test_number','test_originator','analytically_derived_fe?','test_procedure_description','test_fuel_type_description','test_category','aftertreatment_device_cd','aftertreatment_device_desc','police_-_emergency_vehicle?','averaging_method_cd'],axis=1,inplace=True)
data20['averging_method_desc'].value_counts()
data20['fe_unit'].value_counts()
data20.drop(['fe_unit'],axis=1,inplace=True)
data20.columns
cleanup_data20={"averging_method_desc":{"No averaging":0,"Harmonic averaging (1/(Sum(i=1 to n) (FET(i) / WT(i)))":1,"Simple averaging (Sum(i=1 to n) (FET(i)  *  WT(i))) ":2}}
data20.replace(cleanup_data20,inplace=True)
data20.head()


# In[8]:


import seaborn as sns
g=sns.factorplot(data=cdf_20,kind='bar',x="vehicle_type",y="averging_method_desc",palette="dark",alpha=.6,height=4)
#g.despine(left=True)
g.set_axis_labels("vehicle_type","averaging_method_desc")
print("vehicles with type of averging method 2020:\nNo averaging                                                0\nHarmonic averaging (1/(Sum(i=1 to n) (FET(i) / WT(i)))      1\nSimple averaging (Sum(i=1 to n) (FET(i)  *  WT(i)))         2")


# In[6]:


#ANALYSIS


# In[4]:


#1.alternative sources of fuel in 2010
import pandas as pd
anls1=pd.read_csv('Desktop/da/clean_10.csv')
anls2=pd.read_csv('Desktop/da/clean_20.csv')
print("\n1. alternative sources of fuel in 2010\nAnswer:CNG,Cold CO and E85\n")
anls1['test_fuel_type_description'].value_counts()


# In[5]:


print("\n1. alternative sources of fuel in 2020\nAnswer:CNG,Cold CO ,Electricity,Hydrogen 5 and E85\n")
anls2['test_fuel_type_description'].value_counts()


# In[11]:


#UNIQUE MODELS USED ALTERNATIVE SOURCES OF FUEL IN 2010 AND 2020
als1=anls1.query('test_fuel_type_description in ["CNG","E85 (85% Ethanol 15% EPA Unleaded Gasoline)","Cold CO Premium (Tier 2)","Cold CO Regular (CERT)","Cold CO Regular (Tier 2)"]').vehicle_manufacturer_name.nunique()
als2=anls2.query('test_fuel_type_description in ["CNG","E85 (85% Ethanol 15% EPA Unleaded Gasoline)","Cold CO Premium (Tier 2)","Cold CO Regular (Tier 2)","Hydrogen 5","Cold CO Diesel 7-15 ppm Sulfur","Cold CO E10 Premium Gasoline (Tier 3)","Electricity"]').vehicle_manufacturer_name.nunique()


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(['2010','2020'],[als1,als2])
plt.title('Unique models using alternative sources distribution')
plt.xlabel('Year')
plt.ylabel('Number of Unique models')


# In[12]:


als1 , als2


# In[14]:


print("More unique models using alternative sources of fuel in 2020 compared to 2008, look at proportions")
#total unique models each year
total_10=anls1.vehicle_manufacturer_name.nunique()
total_20=anls2.vehicle_manufacturer_name.nunique()
total_10,total_20


# In[16]:


prop_10=als1/total_10
prop_20=als2/total_20
prop_10,prop_20


# In[19]:


plt.bar(['2010','2020'],[prop_10,prop_20])
plt.title('Proportion of Unique models using alternative sources distribution')
plt.xlabel('Year')
plt.ylabel('Proportion of Unique models')
print("More unique models using alternate sources of fuel in 2020 compared to 2010 increased by 30.9%")


# In[42]:


print("2.How much have vehicle drive systems improved in fuel economy?")
#average fuel economy for each dataset
import pandas as pd
anls3=pd.read_csv('Desktop/da/clean_10.csv')
anls4=pd.read_csv('Desktop/da/clean_20.csv')
print("\nyear:2010")
veh10=anls3.groupby('drive_system_description').fe_bag_1.mean()
veh10


# In[43]:


print("\nyear:2020")
veh20=anls4.groupby('drive_system_description').fe_bag_1.mean()
veh20


# In[26]:


#how much have they increased for each drive system type
inc=veh20-veh10
inc


# In[35]:


plt.subplots(figsize=(8,5))
plt.bar(inc.index,inc)
plt.title('Improvements in fuel economy from 2010 to 2020 by vehicle drive system')
plt.xlabel('Drive System type')
plt.ylabel('Increase in average mpg')


# In[47]:


print("3.What are features associated with better fuel economy?")
print("Explore trends between mpg and other features in this dataset, select all vehicles that have the top 50% fuel economy ratings to see characteristics")
#average fuel economy for each dataset
import pandas as pd
anls5=pd.read_csv('Desktop/da/clean_10.csv')
anls6=pd.read_csv('Desktop/da/clean_20.csv')
print("\nyear:2010")
top10=anls5.query('fe_bag_1 > fe_bag_1.mean()')
top10.describe()


# In[46]:


print("\nyear:2020")
top20=anls6.query('fe_bag_1 > fe_bag_1.mean()')
top20.describe()


# In[50]:


print("4.For all the models that were produced in 2010 that are still being produced now, how much has the mpg improved and which vehicle improved the most?")
print("\nsteps:\n1.Create a dataframe , model_mpg that contain the mean mpg values in 2010 and 2020 for each unique model. To do this, group by model and find the mean mpg_2010 and mean mpg_2020 for each.\n2.Create a new column, mpg_change, with change in mpg. Subtract the mean mpg in 2010 from that in 2020 to get change in mpg\n3.Find the vehicle that improved the most. Find the max mpg change, and then use query or indexing to see what model it is!")


# In[16]:


#load datasets
import pandas as pd
anls7=pd.read_csv('Desktop/da/clean_10.csv')
anls8=pd.read_csv('Desktop/da/clean_20.csv')
#merge datasets
new_anls7=anls7.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
new_anls8=anls8.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
#rename 2010 columns
new_anls7.rename(columns=lambda x:  x[:36]+"_2010",inplace=True)
#merge datasets
comb_df=pd.merge(new_anls7,new_anls8,left_on='vehicle_manufacturer_name_2010',right_on='vehicle_manufacturer_name',how='inner')


# In[17]:


comb_df.head()


# In[28]:


#step 1: To do this group by model and find the mean mpg_2010 and mean mpg for each
model_mpg=comb_df.groupby('vehicle_manufacturer_name').mean()[['fe_bag_1_2010','fe_bag_1']]
model_mpg.head()


# In[29]:


#step 2:
model_mpg['mpg_change']=model_mpg['fe_bag_1']-model_mpg['fe_bag_1_2010']
model_mpg.head()


# In[30]:


#step 3:
max_change=model_mpg['mpg_change'].max()
max_change


# In[34]:


print("Mitsubishi Motors Co improved its models the most")
model_mpg[model_mpg['mpg_change']==max_change]


# In[4]:


#checking only for cars
#load datasets
import pandas as pd
anls9=pd.read_csv('Desktop/da/clean_10.csv')
anls10=pd.read_csv('Desktop/da/clean_20.csv')
cars2010=anls9[anls9['vehicle_type']=='Car']
cars2010


# In[5]:


cars2020=anls10[anls10['vehicle_type']=='Car']
cars2020


# In[15]:


car10=cars2010.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
car20=cars2020.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
#rename 2010 columns
car10.rename(columns=lambda x:  x[:36]+"_2010",inplace=True)
#merge datasets
comb_car_df=pd.merge(car10,car20,left_on='represented_test_veh_make_2010',right_on='represented_test_veh_make',how='inner')


# In[16]:


#step 1: To do this group by model and find the mean mpg_2010 and mean mpg for each
car_model_mpg=comb_car_df.groupby('represented_test_veh_make').mean()[['fe_bag_1_2010','fe_bag_1']]
car_model_mpg.head()
#step 2:
car_model_mpg['car_mpg_change']=car_model_mpg['fe_bag_1']-car_model_mpg['fe_bag_1_2010']
car_model_mpg.head()
#step 3:
car_max_change=car_model_mpg['car_mpg_change'].max()
car_max_change
print("Mitsubishi car model improved the most")
car_model_mpg[car_model_mpg['car_mpg_change']==car_max_change]
#Mitsubishi car model improved the most


# In[14]:


#checking only for trucks
#load datasets
import pandas as pd
anls11=pd.read_csv('Desktop/da/clean_10.csv')
anls12=pd.read_csv('Desktop/da/clean_20.csv')
trucks2010=anls11[anls11['vehicle_type']=='Truck']
trucks2010
trucks2020=anls12[anls12['vehicle_type']=='Truck']
trucks2020
truck10=trucks2010.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
truck20=trucks2020.loc[:,['model_year','vehicle_manufacturer_name','represented_test_veh_make','represented_test_veh_model','fe_bag_1']]
#rename 2010 columns
truck10.rename(columns=lambda x:  x[:36]+"_2010",inplace=True)
#merge datasets
comb_truck_df=pd.merge(truck10,truck20,left_on='represented_test_veh_make_2010',right_on='represented_test_veh_make',how='inner')
#step 1: To do this group by model and find the mean mpg_2010 and mean mpg for each
truck_model_mpg=comb_truck_df.groupby('represented_test_veh_make').mean()[['fe_bag_1_2010','fe_bag_1']]
truck_model_mpg.head()
#step 2:
truck_model_mpg['truck_mpg_change']=truck_model_mpg['fe_bag_1']-truck_model_mpg['fe_bag_1_2010']
truck_model_mpg.head()
#step 3:
truck_max_change=truck_model_mpg['truck_mpg_change'].max()
truck_max_change
print("honda truck model improved the most")
truck_model_mpg[truck_model_mpg['truck_mpg_change']==truck_max_change]
#HONDA truck model improved the most


# In[ ]:





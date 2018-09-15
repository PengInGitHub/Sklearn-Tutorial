#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 23:55:24 2018

@author: pengchengliu
"""
#the code
#the code comes from the excellent work of 
#https://github.com/xiaoyusmd/Bj_HousePricePredict

#the task
#analysis the dynamic-changing real easte market of Beijing
#predict the price of 2nd properties
#analsis how the price diffes across districts, type of preperty, size, year, location etc.

#the data
#from two major real estate agency Anjuke.com and lianjia.com
#data resource: web crawler
#https://github.com/xiaoyusmd/Bj_HousePricePredict/tree/master/spiders


#the outline

####EDA
####Feature Engineering
####Modelling





####EDA
###Load Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
root = '/Users/pengchengliu/Documents/GitHub/Sklearn_Tutorial/Beijing Second-Hand Property Price Prediction/'

#Anjuke.com, real estate trade info platform in China
anjuke = pd.read_csv(root+'data/anjuke.csv')

#lianjia.com, Lianjia (Homelink) is a Beijing-based top real estate angency
df = pd.read_csv(root+'data/lianjia.csv')

###dataset structure
anjuke.shape#(3000, 7)
df.shape#(23677, 12)
df.info()
#Elevator has missing 

###visualizaion to single variable

##Elevator
df['Elevator'].value_counts(normalize=True)
#var contains wrong values
#record wrong values
df['Elevator_wrong_value'] = 0
df.loc[(df['Elevator']!='有电梯')&(df['Elevator']!='无电梯')&(df['Elevator'].notnull()) , 'Elevator_wrong_value'] = 1
df['Elevator_wrong_value'].value_counts()

#remove wrong values
df['Elevator'] = df.loc[(df['Elevator']=='有电梯')|(df['Elevator']=='无电梯'), 'Elevator']

#impute missing values
#in Beijing, a building is highly likely to have elevator if there are more than 6 floors
df.loc[(df['Floor']>6)&(df['Elevator'].isnull()) , 'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6)&(df['Elevator'].isnull()) , 'Elevator'] = '无电梯'
df.loc[df['Elevator']=='有电梯' , 'Elevator'] = 'Yes'
df.loc[df['Elevator']=='无电梯' , 'Elevator'] = 'No'
df['Elevator'].value_counts(normalize=True)#

#visualise
f, [ax1, ax2] = plt.subplots(1, 2, figsize=(20,10))
sns.countplot(df['Elevator'], ax=ax1)
ax1.set_title('Comparison in Num of If Elevator Exists')
ax1.set_xlabel('If Elevator Exists')
ax1.set_ylabel('Count')

sns.barplot(x='Elevator', y='Price', data=df, ax=ax2)
ax2.set_title('Comparison in Price of If Elevator Exists')
ax2.set_xlabel('If Elevator Exists')
ax2.set_ylabel('Price')
plt.show()








list(df.columns.values)











































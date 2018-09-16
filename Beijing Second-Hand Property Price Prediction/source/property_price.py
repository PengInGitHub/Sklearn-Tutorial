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

import sys
reload(sys)
sys.setdefaultencoding("utf-8")



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
df.describe()
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

###Floor
##visualise
#use f, ax1 = plt.subplots(figsize=(20,5))
#use sns.countplot(df['var'], ax=ax1)
f, ax1 = plt.subplots(figsize=(20,5))
sns.countplot(df['Floor'], ax=ax1)
ax1.set_title('Count of Real Estate Per Floor', fontsize=15)
ax1.set_xlabel('Floor')
ax1.set_ylabel('Count of Real Estate')
plt.show()
#this feature depends highly on region and culture
#6, 7 and 8 may be prefered in China in general
#this var is significant but quite complex 
help(plt.subplots)

###Layout
f, ax1 = plt.subplots(figsize=(20, 20))
sns.countplot(y='Layout', data=df, ax=ax1)
ax1.set_title('Count of Real Estate Per Layout', fontsize=15)
ax1.set_xlabel('Count of Real Estate')
ax1.set_ylabel('Layout')
plt.show()

###Region

##add new feature
#avg price: price per square
df['PerPrice'] = df['Price']/df['Size']
#check new var
df.head(2)
df_house_count = df.groupby('Region')['Price'].count().sort_values(ascending=False).to_frame().reset_index()
df_house_mean = df.groupby('Region')['PerPrice'].mean().sort_values(ascending=False).to_frame().reset_index()

f, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(20,15))
sns.barplot(x='Region', y='PerPrice', palette='Blues_d', data=df_house_mean, ax=ax1)
ax1.set_title('Comparison of Second-Hand Property Price Per Square By Region', fontsize=15)
ax1.set_xlabel('Region')
ax1.set_ylabel('Price Per Square')

sns.barplot(x='Region', y='Price', palette="Greens_d", data=df_house_count, ax=ax2)
ax2.set_title('Comparison of Second-Hand Property Quantity By Region', fontsize=15)
ax2.set_xlabel('Region')
ax2.set_ylabel('Quantity')

sns.boxplot(x='Region', y='Price', data=df, ax=ax3)
ax3.set_title('Comparison of Second-Hand Property Total Value By Region', fontsize=15)
ax3.set_xlabel('Region')
ax3.set_ylabel('Total Value')

plt.show()


list(df.columns.values)











































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



#################################
#            1.EDA              #
#################################
###Load Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


##by Region another version

# 对二手房区域分组对比二手房数量和每平米房价# 对二手房区 
df['PriceMs'] = df['Price']/df['Size']

df_house_count = df.groupby('Region')['Price'].count().sort_values(ascending=False)
df_house_mean = df.groupby('Region')['PriceMs'].mean().sort_values(ascending=False)

#another way
#this won't work since Price is numerical
#df.groupby('Region').Price.value_counts(normalize=True)
df_house_count.index
#设置X轴刻度标签：
def auto_xtricks(rects,xticks):
    x = []
    for rect in rects:
        x.append(rect.get_x() + rect.get_width()/2)
    x = tuple(x)
    plt.xticks(x,xticks)

#设置数据标签：
def auto_tag(rects, data = None, offset = [0,0], size=14):
    for rect in rects:
        try:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2.4, 1.01*height, '%s' % int(height), fontsize=size)
        except AttributeError:
            x = range(len(data))
            y = data.values
            for i in range(len(x)):
                plt.text(x[i]+offset[0],y[i]+0.05+offset[1],y[i], fontsize='14')
                
def auto_tag_float(rects, data = None, offset = [0,0], size=14):
    for rect in rects:
        try:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2.4, 1.01*height, '%s' % round(float(height),1), fontsize=size)
        except AttributeError:
            x = range(len(data))
            y = data.values
            for i in range(len(x)):
                plt.text(x[i]+offset[0],y[i]+0.05+offset[1],y[i], fontsize='14')

    
plt.figure(figsize=(20,10))
plt.rc('font', family='SimHei', size=13)
plt.style.use('ggplot')

# 各区域二手房数量对比
plt.subplot(212)
plt.title(u'各区域二手房数量对比', fontsize=30)
plt.ylabel(u'二手房总数量（单位：间）', fontsize=15)
rect1 = plt.bar(np.arange(len(df_house_count.index)), df_house_count.values, color='c')
auto_xtricks(rect1, df_house_count.index)
auto_tag(rect1, offset=[-1,0])

# 各区域二手房平均价格对比
plt.subplot(211)
plt.title(u'各区域二手房平均价格对比', fontsize=20)
plt.ylabel(u'二手房平均价格（单位：万/平米）', fontsize=15)
rect2 = plt.bar(np.arange(len(df_house_mean.index)), df_house_mean.values, color='c')
auto_xtricks(rect2, df_house_mean.index)
auto_tag_float(rect2, offset=[-1,0])

plt.figure(figsize=(10,10))
plt.title(u'各区域二手房数量百分比', fontsize=18)
explode=[0]*len(df_house_count)
explode[0] = 0.2
plt.pie(df_house_count, radius=3, autopct='%1.f%%', shadow=True, labels=df_house_count.index, explode=explode)
plt.axis('equal')

plt.show()



#print statistics
df_house_count
df_house_mean


###Renovation
##Price by renovation status
#raw has only few values, but high average price, due to outliers
100*df['Renovation'].value_counts(normalize=True)
f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(df['Renovation'], ax=ax1)
sns.barplot(x='Renovation', y='Price', data=df, ax=ax2)
sns.boxplot(x='Renovation',y='Price',data=df, ax=ax3)
plt.show()


###Size

#distribution of size
#use sns.distplot, sns.kdeplot
#long-tail distribution

f, [ax1,ax2] = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['Size'], bins=20, ax=ax1, color='r')
sns.kdeplot(df['Size'], ax=ax1, shade=True)

#relationship btw price and size
#use sns.regplot()
sns.regplot(x='Size', y='Price', data=df, ax=ax2)
plt.show()

#observe outliers from plot
#inspect outlier
df.loc[df['Size']<10]
df.loc[df['Size']>1000]
#remove outliers
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]


###Year
##use sns.FacetGrid(), parameter: palette='seismic'
grid = sns.FacetGrid(df, row='Elevator', col='Renovation', palette='seismic', size=4)
grid.map(plt.scatter, 'Year', 'Price')


#################################
#     2.Feature Engineering     #
#################################
#overall strategy
#create new features and delete old useless features, do one-hot encoding 

###Layout
df['Layout'].value_counts()
#values in the format of x房间x卫, such as
"""
2房间1卫      170
1房间1卫      146
3房间1卫      116
"""
#should be removed, b/c they are for commerical estate
#use regular expression
df =  df[df['Layout'].astype(str).str.contains('室')]
#the code below does not work
#df = df.loc[df['Layout'].str.extract('^\d(.*?)\d.*?') == '室']

df['Layout_room_num'] = df['Layout'].str.extract('(^\d).*', expand=False).astype('int64')
df['Layout_hall_num'] = df['Layout'].str.extract('^\d.*?(\d).*', expand=False).astype('int64')

###Year
#discretization
#pandas quartile cut
df['Year'] = pd.qcut(df['Year'], 8).astype('object')


###Direction
df['Direction'].value_counts()

def direct_func(x):
    #input x, an instance in col df['Direction']
    #output a str, depends on the lenght 
    
    #sanity check
    if not isinstance(x,str):
        raise TypeError 
    x = x.strip()
    x_len = len(x)
    
    #remove all cases with repeated directions
    #such as 东东南
    x_unique = pd.unique([y for y in x])
    if x_len != len(x_unique):
        return 'no' # would be removed later
    
    #two char cases
    if (x_len == 2)&(x not in d_list_two):
        #return swap to match pattern in d_list_two
        return x[1]+x[0]

    #three char cases
    elif (x_len == 3)&(x not in d_list_three):
        for n in d_list_three:
            if (x_unique[0] in n)&(x_unique[1] in n)&(x_unique[2] in n):
                return n
    
    #four char cases
    elif (x_len == 4)&(x not in d_list_four):
        return d_list_four[0]
        
    else:
        return x
    



d_list_one = ['东','西','南','北']
d_list_two = ['东西','东南','东北','西南','西北','南北']
d_list_three = ['东西南','东西北','东南北','西南北']
d_list_four = ['东西南北']
df['Direction'] = df['Direction'].apply(direct_func)
#df.drop('Direction_new', axis=1, inplace=True)
df.isnull().sum() #check missing value
df = df.loc[(df['Direction']!='no')&(df['Direction']!='nan')] 


###Layout
df['Layout_total_num'] = df['Layout_room_num'] + df['Layout_hall_num']
df['Size_room_ratio'] = df['Size']/df['Layout_total_num']


###Remove columns
df.drop(['Layout','PerPrice','Garden', 'District'], axis=1, inplace=True)


###One Hot Encoding
#use pandas get_dummies()
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_col = [col for col in df.columns if df[col].dtype=='object']
    df = pd.get_dummies(df, columns=categorical_col, dummy_na=nan_as_category)
    new_cols = [col for col in df.columns if col not in original_columns]
    return df, new_cols
 
#execute
df, df_cat = one_hot_encoder(df)    
     
#corr heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), linewidth=0.1, vmax=1.0, square=True,
           cmap=colormap, linecolor='white', annot=True)

#calculate correaltion matrix
#52*52
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

#drop
df.drop(to_drop, axis=1, inplace=True)
#only 50 features left



####################################
#           3.Modelling            #
####################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import visuals as vs

features = df.drop('Price', axis=1)
prices = df['Price']

###to np array
features, prices = np.array(features), np.array(prices)
features_train, features_test, prices_train, prices_test = train_test_split(features, prices, test_size=0.2, random_state=0)

###fit model

##build performance_metric
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

##build grid
def build_grid():
    regressor = DecisionTreeRegressor() 
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_func = make_scorer(performance_metric)
    cross_validator = KFold(10, shuffle=True)
    grid = GridSearchCV(estimator = regressor, param_grid=params, 
                        scoring=scoring_func, cv=cross_validator)
    return grid

##fit model with the grid
def fit_model(x, y):
    
    #build the grid searching via GridSearchCV
    grid = build_grid()
    #use the grid searching to fit 
    grid = grid.fit(x, y)
    return grid.best_estimator_


###model evaluation
vs.ModelLearning(features_train, features_test)
vs.ModelComplexity(features_train, prices_train)

optimal_reg1 = fit_model(features_train, prices_train)

optimal_reg1.get_params()['max_depth']#9

predicted_value = optimal_reg1.predict(features_test)
r2 = performance_metric(prices_test, predicted_value)
r2
#0.575754796918848
#under fitting

list(df.columns.values)











































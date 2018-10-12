# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:21:38 2018

@author: keyuemei
"""


from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data file
data0 = pd.read_csv("D://Sunmnet//Employment_trend_predict//data//stu_info_data_5.csv",encoding = 'utf8')

# Data preprocessing
data = data0.drop(['RUXUENIANYUE'],axis = 1)

data["salary"] = data["salary"].map(lambda x: x/12 if (x >= 40000) else x)
data = data.ix[data.salary <= 15000]
data = data.ix[data.salary >= 2000]
data['DANWEIXINGZHI'] = data['DANWEIXINGZHI'].replace(
        "国有企业","国企").replace(
        "机关","国企").replace(
        "部队","国企").replace(
        "其他企业","其他企业").replace(
        "医疗卫生单位","其他企业").replace(
        "中初教育单位","其他企业").replace(
        "高等教育单位","其他企业").replace(
        "三资企业","其他企业").replace(
        "其他事业单位","其他企业").replace(
        "科研设计单位","其他企业").replace(
        "其他","其他").replace(
        "自主创业","其他").replace(
        "自由职业","其他")
data= data.fillna(0.0)
# Box figure
#plt.boxplot(data['salary'])

"""Dummy variable conversion"""
data_dummies = pd.get_dummies(data[:])

"""Data normalization"""
data_dummies_norm = data_dummies.ix[:,'sy_salary':'DANWEIXINGZHI_国企'].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))

data_dummies_all_data = data_dummies[['xh', 'salary']].join(data_dummies_norm)

# The correlation
data_corr = data_dummies_all_data.corr()
data_corr_sorted =data_corr.sort_index(axis = 0,ascending = False,by = 'salary')

data_dummies_all_data = data_dummies_all_data.drop(['by_score', 'XFCJ', 'gk_xf', 'col_0',
       'col_1', 'col_2', 'col_3', 'col_4', 'col_5'],axis = 1)

'''
data_dummies_all_data = data_dummies_all_data.drop(['by_score', 'ZHENGZHIMIANMAO_共产党', 'col_3',
       'ZHENGZHIMIANMAO_共青团', 'score_min', 'col_7', 'ZHENGZHIMIANMAO_其他',
       'col_2', 'col_6', 'col_5', 'col_4'],axis = 1)
'''

"""split data"""
from sklearn.model_selection import train_test_split
trainData, testData = train_test_split(data_dummies_all_data, test_size=0.2,random_state=888)

train_cols = trainData.columns[2:]
test_cols = testData.columns[2:]

trainData_X = np.array(trainData[train_cols])
trainData_Y = np.array(trainData[["salary"]])

testData_X = np.array(testData[test_cols])
testData_Y = np.array(testData[["salary"]])

""" 线性回归 """
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def model_build(model = linear_model,
                trainX=trainData_X,
                trainY=trainData_Y,
                testX=testData_X):
    linear = model.LinearRegression()
    linear.fit(trainX, trainY)
    print(linear.score(trainX,trainY))
    #a, b = linear.coef_, linear.intercept_
    test_predict = linear.predict(testX)
    return test_predict

"""Model evaluation"""
def MAPE(y_pre, y_true): 
    Mape = np.mean(np.abs(y_pre - y_true)/np.abs(y_true)) 
    print(Mape) # Calculate the error rate of average absolute value
    return Mape

test_predict = model_build(linear_model,trainData_X,trainData_Y,testData_X)
test_mape = MAPE(test_predict,testData_Y)  #0.1296





"""
# make model
linear = linear_model.LinearRegression()
                                                                                                                                                                                                                                                                                                                                                                    
# fit
linear.fit(trainData_X, trainData_Y)
linear.score(trainData_X,trainData_Y)

#Equation coefficient and Intercept 系数、截距
a, b = linear.coef_, linear.intercept_

# 根据predict方法预测的值z
#print(linear.predict(testData_X))
test_predict = linear.predict(testData_X)
"""




















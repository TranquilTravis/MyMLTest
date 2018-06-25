#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:59:01 2018

@author: dq
"""

import pandas as pd

# import data to dataFrame
df = pd.read_csv('atec_anti_fraud_train.csv', sep=',', header=0)

#%% print label information
print("number of label 0 is: {0}".format(sum(df.iloc[:,1]==0)))
print("number of label 1 is: {0}".format(sum(df.iloc[:,1]==1)))
print("number of label -1 is: {0}".format(sum(df.iloc[:,1]==-1)))

# label list index
labelZero = df.index[df['label'] == 0].tolist()
labelOne = df.index[df['label'] == 1].tolist()
labelMinus = df.index[df['label'] == -1].tolist()

### construct only labeled data
dfLabel = df.loc[df['label'] != -1]

#%% correlation between features and to labels
########## ls current directory
from subprocess import check_output
print(check_output(["ls", "."]).decode("utf8"))

#Lets start by plotting a heatmap to determine if any variables are correlated
import matplotlib.pyplot as plt
import seaborn as sns
# use all data, because label is irrelavant here
dataCorr=df.iloc[:,3:].corr()
# use only labeled data
corrToLabel = [dfLabel['label'].corr(dfLabel.iloc[:,i]) for i in range(3,300)]
plt.figure()
ax = sns.heatmap(dataCorr)
plt.show()
#plt.gcf().clear()

#plt.bar(dfLabel.columns[3:].tolist(),corrToLabel)
#plt.ylabel('Relation to label')

#%%%%%%%%%%%%%%%%%%% missing data ####################
missing_df = dfLabel.iloc[:,3:].isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
#missing_df = missing_df.loc[missing_df['missing_count']>0]
#missing_df = missing_df.sort_values(by='missing_count')

########### relationship between missing data and label ##############
############## decide how to deal with missing data ######################
missAndCorr=pd.DataFrame(missing_df)
missAndCorr['correlation'] = corrToLabel

# visualization
plt.bar(missAndCorr['column_name'].tolist(),missAndCorr['missing_count'].tolist())
plt.ylabel('Count of missing values')

## another way
#import numpy as np
#missAndCorr = missAndCorr.sort_values(by='missing_count')
#ind = np.arange(missAndCorr.shape[0])
#width = 0.9
#fig, ax = plt.subplots(figsize=(12,18))
#rects = ax.barh(ind, missAndCorr.missing_count.values, color='blue')
#ax.set_yticks(ind)
#ax.set_yticklabels(missAndCorr.column_name.values, rotation='horizontal')
#ax.set_xlabel("Count of missing values")
#ax.set_title("Number of missing values in each column")
#plt.show()

#%%% if 'missing' is correlated to labels
#### first convert label 0 to -1, then convert non-missing data to -1, and missing data to 1
dfLabelConvert = dfLabel.drop(columns=['id','date'])
dfLabelConvert.iloc[:,1:].where(dfLabelConvert.isnull(),-1,inplace=True)
dfLabelConvert.fillna(1,inplace=True)
# the pearson's correlationship between 'missing' and label
corrMissing = [dfLabelConvert['label'].corr(dfLabelConvert.iloc[:,i]) for i in range(1,dfLabelConvert.shape[1])]
#from scipy.stats import pearsonr
#corrMissing = [pearsonr(dfLabelConvert['label'], dfLabelConvert.iloc[:,i])[0] for i in range(1,dfLabelConvert.shape[1])]
plt.figure()
plt.bar(dfLabelConvert.columns[1:].tolist(),corrMissing)
plt.title('pearsons correlation to label')

# use all data, because label is irrelavant here
plt.figure()
missCorr=dfLabelConvert.iloc[:,1:].corr()
ax = sns.heatmap(missCorr)
plt.show()

#from scipy.stats import pearsonr
#corrMissing = [pearsonr(dfLabelConvert['label'], dfLabelConvert.iloc[:,i])[0] for i in range(1,dfLabelConvert.shape[1])]


#%% clean the data
import numpy as np
from sklearn import preprocessing
# split labeled data into training and testing sets
mask= np.random.rand(len(dfLabel)) < 0.8
trainData=dfLabel.iloc[mask,3:].values
trainLabel=dfLabel.iloc[mask,1].values

testData=dfLabel.iloc[~mask,3:].values
testLabel=dfLabel.iloc[~mask,1].values

# simply impute missing data with mean value
from sklearn.preprocessing import Imputer
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
imptrainData = pd.DataFrame(fill_NaN.fit_transform(trainData))
imptestData = pd.DataFrame(fill_NaN.fit_transform(testData))

# normalize features
nortrainData = (imptrainData - imptrainData.mean())/imptrainData.std()
nortestData = (imptestData - imptestData.mean())/imptestData.std()

#%% reduce feature dimension by autoencoder
from autoencoder import encoder

output_train , output_test = encoder(nortrainData,nortestData)


#%%%%%%%%%%%%%%%% use gbm #############
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.grid_search import GridSearchCV   #Performing grid search
from gbmModel import modelfit
from gbmModel import modelvali

#All of the rc settings are stored in a dictionary-like variable called matplotlib.rcParams
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

gbm0 = GradientBoostingClassifier(random_state=10)
#gbm_tuned = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,warm_start=True)
alggbm = modelfit(gbm0, output_train, trainLabel)
#%%
modelvali(alggbm, output_train, trainLabel)

#%%%%%%%%%%%%%%%% use xgboost
from xgboost.sklearn import XGBClassifier
from xgbModel import xgbfit, xgbvali

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

algxgb = xgbfit(xgb1, output_train, trainLabel)

xgbvali(algxgb, output_train, trainLabel)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:02:50 2018

@author: dq
"""
import numpy as np
from sklearn import cross_validation, metrics   #Additional scklearn functions

def modelfit(alg, train, label):
    #Fit the algorithm on the data
    alg.fit(train, label)
    
    return alg

def modelvali(alg, train, label, performCV=True, printFeatureImportance=True, cv_folds=5):
        
    #Predict training set:
    dtrain_predictions = alg.predict(train)
    dtrain_predprob = alg.predict_proba(train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, train, label, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(label, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(label, dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
#    if printFeatureImportance:
#        feat_imp = pd.Series(alg.feature_importances_, train.columns).sort_values(ascending=False)
#        feat_imp.plot(kind='bar', title='Feature Importances')
#        plt.ylabel('Feature Importance Score')
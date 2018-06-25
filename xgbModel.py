#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:00:12 2018

@author: dq
"""

import xgboost as xgb
from sklearn import metrics   #Additional scklearn functions

def xgbfit(alg, train, label, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train.values, label=label.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train, label,eval_metric='auc')
    
    return alg

def xgbvali(alg, train, label):
        
    #Predict training set:
    dtrain_predictions = alg.predict(train)
    dtrain_predprob = alg.predict_proba(train)[:,1]
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train, dtrain_predprob))
                    
#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#    feat_imp.plot(kind='bar', title='Feature Importances')
#    plt.ylabel('Feature Importance Score')
        
    
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:27:55 2020

@author: JTSDellLaptop
"""
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

#function fits, predicts, and evaluates input data / model accuracy using an input classifer
def classify(classifier,data):
    X = data[:,:-1]
    y = data[:,-1]
    # use kfold training with 5 splits
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    auc = list()
    thresholds = list()
    i = 0
    for train_index, test_index in kf.split(X):

        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # predict probabilities
        lr_probs = classifier.predict_proba(X_test)
        # keep probabilities for the positive outcome only (TPR (Sensitivity), FPR (1- Speficity))
        lr_probs = lr_probs[:, 1]
        ns_probs = [0 for _ in range(len(y_test))] # Nominal
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        auc.append(lr_auc)
        # calculate roc curves
        ns_fpr, ns_tpr, ns_thresh = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test, lr_probs)
        gmeans = np.sqrt(lr_tpr * (1-lr_fpr))
        opt_thresh = lr_thresh[np.argmax(gmeans)]
        thresholds.append(opt_thresh)
        
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        print(i)
        i+=1
        
    return auc, thresholds
        
        
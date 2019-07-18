# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:40:18 2019

@author: Karra's
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cat_data=pd.read_csv("D:/ml_github/cat_data.csv") 
cat_data_full_data=pd.read_csv("D:/ml_github/cat_data_full_data.csv") 
cat_data_with_null_data=pd.read_csv("D:/ml_github/cat_data_with_null_data.csv") 

cat_full_data_col=cat_data_full_data.shape[1]
bigdata = pd.concat([cat_data_full_data, cat_data_with_null_data], axis=1) 

#for i in range(0,cat_data_with_null_data.shape[1]):
for i in range(0,2):    
    null_indices=cat_data_with_null_data[cat_data_with_null_data.iloc[:,i].isnull()].index.tolist()
    non_null_indices=cat_data_with_null_data[~cat_data_with_null_data.iloc[:,i].isnull()].index.tolist()
    
    X_train=cat_data_full_data.iloc[non_null_indices]
    X_test=cat_data_full_data.iloc[null_indices]
    
    
    y_train=cat_data_with_null_data.iloc[:,i][non_null_indices]
    
    y_test=cat_data_with_null_data.iloc[:,i][null_indices]
    
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_test)
    
    
    
    cat_data_with_null_data.iloc[:,i][null_indices]=y_pred
  #  cat_data_full_data=pd.concat([cat_data_full_data,cat_data_with_null_data.iloc[:,i]],axis=1)
    from sklearn.model_selection import KFold
    scores = []
    f1_scores=[]
    precision_scores=[]
    recall_scores=[]
    X_train_new=X_train.values
    y_train_new=y_train.values
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X_train_new):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
    
        X_train_, X_test_, y_train_, y_test_ = X_train_new[train_index], X_train_new[test_index], y_train_new[train_index], y_train_new[test_index]
        model.fit(X_train_, y_train_)
        scores.append(model.score(X_test_, y_test_))
        y_pred_test=model.predict(X_test_)
        f1_scores.append(f1_score(y_test_, y_pred_test, average="macro"))
        precision_scores.append(precision_score(y_test_, y_pred_test, average="macro"))
        recall_scores.append(recall_score(y_test_, y_pred_test, average="macro"))
        
    Avg_f1_score=  sum(f1_scores)/len(f1_scores)
    Avg_precision_score= sum(precision_scores)/len(precision_scores)
    Avg_recall_score=  sum(recall_scores)/len(recall_scores)
     
    
    y_pred_train=model.predict(X_train)
    conf_mat = confusion_matrix(y_train, y_pred_train)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
            #    xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    
    print(f1_score(y_train, y_pred_train, average="macro"))
    print(precision_score(y_train, y_pred_train, average="macro"))
    print(recall_score(y_train, y_pred_train, average="macro")) 




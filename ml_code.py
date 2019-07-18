
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:22 2019

@author: Karra's
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("D:/Masters/Sem_2/Machine_Learning/project/ML1/train.csv", sep='\t',engine='python')
train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
train.isna().sum()

null_counts = train.isnull().sum()/len(train)
greater_than_zero_nulls=null_counts[null_counts>0.1]
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(greater_than_zero_nulls))+0.5,greater_than_zero_nulls.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(greater_than_zero_nulls)),greater_than_zero_nulls)
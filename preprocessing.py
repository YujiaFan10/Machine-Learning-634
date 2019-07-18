# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 03:35:22 2019

@author: Karra's
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("D:/Masters/Sem_2/Machine_Learning/project/ML1/train.csv", sep='\t',engine='python')
#train = pd.read_csv("D:/Masters/Sem_2/Machine_Learning/project/ML1/from_sav_data.csv", sep=',',engine='python')
#train = pd.read_csv("D:/Masters/Sem_2/Machine_Learning/project/ML3/ML3AllSites.csv", sep=',',engine='python')
train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
train = train.replace('\.+', np.nan, regex=True)
train.isna().sum()
train = train.drop_duplicates()
#######not needed#############3
#null_counts = train.isnull().sum()/len(train)
#greater_than_zero_nulls=null_counts[null_counts>0.1]
#plt.figure(figsize=(16,8))
#plt.xticks(np.arange(len(greater_than_zero_nulls))+0.5,greater_than_zero_nulls.index,rotation='vertical')
#plt.ylabel('fraction of rows with missing data')
#plt.bar(np.arange(len(greater_than_zero_nulls)),greater_than_zero_nulls)



col_uni_val={}
for i in train.columns:
    col_uni_val[i] = len(train[i].unique())

d = [k for k, v in col_uni_val.items() if v == 1]

train.drop(d, axis=1,inplace=True)

#'beginlocaltime','user_agent','study_url','moneyetnicitya','expcomments','study_name',
train.drop(['moneyethnicitya','ContactGroup','text','user_id','previous_session_id','moneyethnicityb','text','feedback','diseaseframinga','diseaseframingb','imagineddescribe','session_id','session_date','last_update_date','session_last_update_date','creation_date','session_creation_date','gainlossgroup','gainlossDV'],axis=1,inplace=True)

unwanted = train.columns[train.columns.str.startswith(('task','iat','sys','scales','exp','flagsupplement'))]

train.drop(unwanted, axis=1, inplace=True)

#,'expgender','diseaseframinga','diseaseframingb',
cat_cols=['referrer','compensation','recruitment','separatedornot','sample','sunkgroup','partgender',

'anch1group','anch2group','flagfilter','anch3group','anch4group','gambfalgroup',
'reciprocitygroup','reciprocityus','flagGroup','MoneyGroup',
'allowedforbiddenGroup','allowedforbidden','quoteGroup','citizenship',
'flagtimeestimate1','flagtimeestimate2','flagtimeestimate3',
'flagtimeestimate4','nativelang','nativelang2','noflagtimeestimate1',
'noflagtimeestimate2','noflagtimeestimate3','noflagtimeestimate4',
'omdimc3','politicalid','race','reciprocityothera','reciprocityotherb',
'reciprocityusa','reciprocityusb','sex','previous_session_schema',
'us_or_international', 'lab_or_online','citizenship2','religion',
'mturk.Submitted.PaymentReq','mturk.non.US','filter_$','o1','o2','o3',
'o4','o5','o6','o7','o8','o9','o10','o11','allowedforbiddena','allowedforbiddenb']

for i in cat_cols:
    train[i] = pd.factorize(train[i])[0]
    train[i].replace(-1,np.nan,inplace=True)
    
    
#train[~pd.isnull(train).any(axis=1)]    
new_train=train.loc[:,pd.isnull(train).sum()<=0] 


new_train_2=train.loc[:,pd.isnull(train).sum()>0]

cont_columns=['age','anchoring1','anchoring2','anchoring3','anchoring4','Ranchori','RAN001','RAN002','RAN003',
              'Ranch1','Ranch2','Ranch3','Ranch4','anchoring1akm','anchoring1bkm','anchoring3ameter','anchoring3bmeter',
              'lat11','lat12','lat21','lat22']

cat_data=train.drop(cont_columns, axis=1)

cat_data_full_data=train.loc[:,pd.isnull(train).sum()<=0] 


cat_data_with_null_data=train.loc[:,pd.isnull(train).sum()>0]

cat_data_full_data.to_csv("D:/ml_github/cat_data_full_data.csv", index=False)
cat_data_with_null_data.to_csv("D:/ml_github/cat_data_with_null_data.csv", index=False)

cat_data.to_csv("D:/ml_github/cat_data.csv", index=False)   
#cat_data=cat_data.apply(pd.to_numeric,errors='coerce')

#cat_data=cat_data.fillna(-999).astype(int)

#cat_data.replace(-999, np.nan, inplace=True)

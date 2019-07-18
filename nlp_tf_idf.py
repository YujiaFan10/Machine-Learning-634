# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:47:19 2019

@author: Karra's
"""
from collections import Counter
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
#allsentences=train['highpower'].astype(str).tolist()
stop = set(stopwords.words('english')) 
print(stop)

import re
import nltk
temp =[]
snow = nltk.stem.SnowballStemmer('english')

import numpy
#cat_data=pd.read_csv("D:/ml_github/cat_data.csv")
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

train.drop(['numparticipants_actual'],axis=1,inplace=True)

final_X=train['imagineddescribe'].astype(str)

#final_X=train['highower'].astype(str)
for sentence in final_X:
    sentence = sentence.lower()                 # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations
    
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords
    temp.append(words)
    
final_X = temp 

sent = []
for row in final_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

allsentences = sent


DF = {}
for i in range(len(allsentences)):
    tokens = allsentences[i]
    for w in tokens.split():
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
            
for i in DF:
    DF[i]=len(DF[i])
    
total_vocab=[x for x in DF]
print(total_vocab)

tf_idf = {}
for i in range(len(allsentences)):
    tokens = allsentences[i].split()
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/(len(tokens))
        df = DF[token]
        idf = np.log(len(allsentences)/(df+1))
        tf_idf[i, token] = tf*idf
        
        
# Document Vectorization
D = np.zeros((len(allsentences), len(total_vocab)))
for i in tf_idf:
    ind = total_vocab.index(i[1])
    D[i[0]][ind] = tf_idf[i]
    
    
nlp_df=pd.DataFrame(D)    
#bigdata = pd.concat([train, nlp_df], ignore_index=True)
bigdata = pd.concat([train, nlp_df], axis=1) 

from sklearn.metrics.pairwise import cosine_similarity
cat_data=pd.read_csv("D:/ml_github/cat_data.csv") 
cat_data=cat_data.fillna(0) 
#train_=train.astype(int)
cos_ = cosine_similarity(cat_data)
np.fill_diagonal(cos_, -100)
m=np.argmax(cos_,axis=1)
x=train.loc[m,'imagineddescribe']

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:35:58 2019

@author: Karra's
"""



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

allsentences=train['imagineddescribe'].astype(str).tolist()
import re
def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))
 
def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def generate_bow(allsentences):  
    l=[]
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                if word == w: 
                    bag_vector[i] += 1
                    
        print("{0}\n{1}\n".format(sentence,numpy.array(bag_vector)))
        l.append(numpy.array(bag_vector))
    return l
n=generate_bow(allsentences)        

nlp_df=pd.DataFrame(list(map(np.ravel, n)))


bigdata = pd.concat([train, nlp_df], axis=1) 

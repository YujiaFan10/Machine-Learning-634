# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:08:11 2019

@author: Karra's
"""

#cat_data=cat_data.apply(pd.to_numeric,errors='coerce')

#cat_data=cat_data.fillna(-999).astype(int)

#cat_data.replace(-999, np.nan, inplace=True)

from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer

stop = set(stopwords.words('english')) 
print(stop)

import re
import nltk
temp =[]
snow = nltk.stem.SnowballStemmer('english')
final_X=train['highower'].astype(str)
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

final_X = sent


splitted = []
w2v_data = final_X
for row in w2v_data: 
    splitted.append([word for word in row.split()])     #splitting word
    
from gensim.models import Word2Vec 
train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)

tf_w_data = final_X

from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=5000)
tf_idf_data = tf_idf.fit_transform(tf_w_data)

tf_w_data = []
tf_idf_data = tf_idf_data.toarray()
i = 0


for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    if tf_idf_sum==0:
        tf_idf_sum=.01
    #print(vec,tf_idf_sum)    
    vec[:] = [(float)(1/tf_idf_sum) * x for x in vec]
    #vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1
    if i==tf_idf_data.shape[0]:
        break

v=pd.DataFrame(list(map(np.ravel, tf_w_data)))

print(tf_w_data[1])

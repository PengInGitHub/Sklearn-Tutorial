#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:03:24 2018

@author: pengchengliu
"""

####Loading the 20 newsgroups dataset
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, random_state=46, shuffle=True)
type(twenty_train)
#sklearn.utils.Bunch
twenty_train.target_names
type(twenty_train.data)
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[:3])

#get back the category names
for t in twenty_train.target[:10]:
    print twenty_train.target_names[t]

#high dimensional sparse data
#save memory by sorting only non-zero parts
    
####Extracting features from text files    
    
##Tokenizing text with scikit-learn¶
    
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_trans_count = count_vect.fit_transform(twenty_train.data)
X_trans_count.shape
#(2257, 35788)
X_trans_count

##From occurrences to frequencies

#term frequency
#Term Frequency times Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_trans_count)
X_train_tfidf.shape
type(X_train_tfidf)
#scipy.sparse.csr.csr_matrix


####Training a classifier
#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)




























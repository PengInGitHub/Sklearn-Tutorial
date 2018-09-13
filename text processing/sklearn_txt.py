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



####Building a pipeline

#vectorizer => transformer=> classifier
from sklearn.pipeline import Pipeline
    
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])    

#error made here: 
text_clf.fit(twenty_train.data, twenty_train.target)  
#evaluate the predictive accuracy
#get test data
twenty_test = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
import numpy as np
np.mean(predicted == twenty_test.target)
#average accuracy 83%    

#further performance analysis
from sklearn import metrics
print metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names)


####Parameter tuning using grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
idx = gs_clf.predict(['God is a lover'])[0]
twenty_train.target_names[idx]
gs_clf.best_params_



from sklearn.linear_model import LogisticRegression

text_clf_3 = Pipeline([('vect',CountVectorizer(decode_error='ignore')),
                    ('tfidf',TfidfTransformer()),
                    ('clf',LogisticRegression()),
                    ])

text_clf_3 = text_clf_3.fit(twenty_train.data,twenty_train.target)
predicted = text_clf_3.predict(docs_test)

np.mean(predicted == twenty_test.target)


























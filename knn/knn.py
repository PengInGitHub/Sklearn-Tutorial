


###########################################
#            1.Preprocessing              #
###########################################
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

##load data
iris = load_iris()
data_X, data_y = iris.data, iris.target 

##check shape
data_X.shape#(150,4)
data_y.shape#(150,)

##split data
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=.2)


###########################################
#         2.Modeling & Prediction         #
###########################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(train_X, train_y)
predict = knn.predict(test_X)



model = GaussianNB()
model = LogisticRegression()
model = DecisionTreeClassifier()
model = SVC()
model.fit(train_X, train_y)
predict = model.predict(test_X)

###########################################
#         3.Prediction Evaluation         #
###########################################
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

fpr, tpr, thresholds = metrics.roc_curve(test_y, predict, pos_label=1)
metrics.auc(fpr, tpr)
#KNN   0.6164772727272726
#NB    0.6164772727272726
#LR    0.6420454545454545
#Tree  0.6164772727272726

precision_recall_fscore_support(test_y, predict)
metrics.classification_report(test_y, predict)
metrics.confusion_matrix(test_y, predict)

###Hyper Parameter Tuning
##GridSearchCV
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

model = Ridge()
model = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
model.fit(train_X, train_y)
model.best_score_
#0.9181257968609238
#best alpha
model.best_estimator_.alpha
#1


###########################################
#           4.Model Persistence           #
###########################################

##save prediction to pickle
import joblib
joblib.dump(knn, '/tmp/knn.pkl')
model = joblib.load('/tmp/knn.pkl')




















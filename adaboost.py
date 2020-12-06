# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:10:54 2020

@author: user
"""


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

##############################################################################

# 1. 데이터 불러오기.

diabetes = pd.read_csv('data/diabetes_csv.csv', header=0)
liver = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv', header=0)
ionosphere = pd.read_table('data/ionosphere_data.txt', sep=',', header=None).iloc[0, :]
ionos = []
for i in ionosphere:
    ionos.append(i.split(','))
ionosphere = pd.DataFrame(ionos)
del ionos

##############################################################################

# 2. 데이터 전처리
# 데이터 label 값 {1, -1} 수정.
# 범주형 변수 값 변경.
# np.array 로 바꾸기 X,y.
# nan 행 제거.

# diabetes dataset
new_class = []
for i in range(len(diabetes)):
    if diabetes.loc[i, 'class'] == 'tested_positive':
        new_class.append(1)
    else:
        new_class.append(-1)
diabetes['class'] = new_class

# liver dataset
new_class = []
gender_class = []
for i in range(len(liver)):
    if liver.loc[i, 'liver patient'] == 1:
        new_class.append(1)
    else:
        new_class.append(-1)

    if liver.loc[i, 'gender'] == 'Male':
        gender_class.append(1)
    else:
        gender_class.append(0)

liver['liver patient'] = new_class
liver['gender'] = gender_class
liver = liver[liver['?A/G Ratio Albumin and Globulin Ratio'].apply(lambda x: not math.isnan(x))]

# ionosphere dataset
new_class = []
for i in range(len(ionosphere)):
    if ionosphere.loc[i, 34] == 'g\n':
        new_class.append(1)
    else:
        new_class.append(-1)
ionosphere[34] = new_class


# diabetes
X = np.array(diabetes.iloc[:, :-1])
y = np.array(diabetes.iloc[:, -1])

# # liver
# X = np.array(liver.iloc[:, :-1])
# y = np.array(liver.iloc[:, -1])

# # ionosophere
# X = np.array(ionosphere.iloc[:, :-1])
# y = np.array(ionosphere.iloc[:, -1])


##############################################################################

# 3. Adaboost 구현.
class AdaBoost:

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None
        self.ht = None

    def _check_X_y(self, X, y):
        assert set(y) == {1, -1}
        return X, y

def fit(self, X: np.ndarray, y: np.ndarray, iters: int):

    X, y = self._check_X_y(X, y)
    n = X.shape[0]

    # init numpy array
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)
    self.ht = np.zeros(shape=(iters, n))
    
    
    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n


    for t in range(iters):

        # fit weak learner
        curr_sample_weights = self.sample_weights[t]  # current sample weight
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump = stump.fit(X, y, sample_weight=curr_sample_weights)

        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)  # y_i
        err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
        stump_weight = np.log((1 - err) / err) / 2  # alpha minimizing misclassified sample's w

        # update sample weights
        new_sample_weights = (
            curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )

        # normalization new sample weight
        new_sample_weights /= new_sample_weights.sum()


        # If not final iteration, update sample weights for t+1
        if t+1 < iters:
            self.sample_weights[t+1] = new_sample_weights


        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

    return self


def predict(self, X):

    stump_preds = np.array([stump.predict(X) for stump in self.stumps])
    return np.sign(np.dot(self.stump_weights, stump_preds))


##############################################################################

result=[]
for rs in [1,2,3,4,5]:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=False,
                                                        random_state=rs)
    AdaBoost.fit = fit
    AdaBoost.predict = predict

    clf = AdaBoost().fit(X_train, y_train, iters=10)

    test_err = (clf.predict(X_test) != y_test).mean()
    result.append(test_err)
np.mean(result)
    
    
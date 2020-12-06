# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:28:56 2020

@author: user
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:10:54 2020

@author: user
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import tqdm 

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
        self.hj = None

    def _check_X_y(self, X, y):
        assert set(y) == {1, -1}
        return X, y

# iters = 10
# eps = 0.05

def fit(self, X: np.ndarray, y: np.ndarray, iters: int, eps=0.03):

    X, y = self._check_X_y(X, y)
    n = X.shape[0]


    # init numpy array
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)
    self.hj = []
    self.pre_e = None
    self.pre_y = None
    self.pre_st_w = None

    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n


    for t in range(iters):
        curr_sample_weights = self.sample_weights[t]  # current sample weight

        if t == 0:  # 이전 err 가 없는 첫번째 경우.
            errs = {}
            for i in range(X.shape[1]):
                errs[i] = []
                # fit weak learner
                stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
                X_single = X[:, i].reshape((-1, 1))
                stump = stump.fit(X_single, y, sample_weight = curr_sample_weights)
                
                stump_pred = stump.predict(X_single)  # y_i
                err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
                errs[i].append(err)
                errs[i].append(stump_pred)
    
            # 1.2 find lowest error.
            lowest_error = sorted(errs.items(), key=lambda x: x[1][0], reverse=False)[0]
            not_lowest_error = sorted(errs.items(), key=lambda x: x[1][0], reverse=False)[1:]
    
            stump_pred = errs[lowest_error[0]][1]  # predict of lowest error
            err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
            stump_weight = np.log((1 - err) / err) / 2  # alpha minimizing misclassified sample's w
            
            X_single = X[:, lowest_error[0]].reshape((-1, 1))
            stump = stump.fit(X_single, y, sample_weight = curr_sample_weights)
            self.hj.append(lowest_error[0]) # weak classifier ht 
            
            
        else:
            # 1. Create the candidate classifiers set, C
            # 1.1 fit weak learner each feature.
            # errs 는 각 feature의 error와 예측값을 가지고 있음.
            errs = {}
            for i in range(X.shape[1]):
                errs[i] = []
                # fit weak learner
                stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
                X_single = X[:, i].reshape((-1, 1))
                stump = stump.fit(X_single, y, sample_weight = curr_sample_weights)
                
                stump_pred = stump.predict(X_single)  # y_i
                err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
                errs[i].append(err)
                errs[i].append(stump_pred)

            # 1.2 find lowest error.
            lowest_error = sorted(errs.items(), key=lambda x: x[1][0], reverse=False)[0]
            not_lowest_error = sorted(errs.items(), key=lambda x: x[1][0], reverse=False)[1:]

            # 2. Calculate diversity between the ensemble classifier
            # in the previous cycle and each candidate classifier.
            # set C
            C = set()
            for i in not_lowest_error:
                if abs(lowest_error[1][0]-i[1][0]) < eps:
                    C.add(i[0])

            # find largest diversity among the candidate classifiers.
            if len(C) > 0:
                D = {}
                for c in C:
                    stump_pred = errs[c][1]
                    d = sum(sum([(stump_pred != self.pre_y)])) / X.shape[0]
                    D[c] = d

                largest_diversity_c = sorted(D.items(), key=lambda x: x[1], reverse=True)[0]

                stump_pred = errs[largest_diversity_c[0]][1]  # predict of lowest error
                err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
                stump_weight = np.log((1 - err) / err) / 2  # alpha minimizing misclassified sample's w

                self.hj.append(largest_diversity_c[0])
            
                # diversity 가 가장 높은 hj 를 ht 로 사용. 
                X_single = X[:, largest_diversity_c[0]].reshape((-1, 1))
                stump = stump.fit(X_single, y, sample_weight = curr_sample_weights) #ht
                
            else:
                stump_pred = errs[lowest_error[0]][1]  # predict of lowest error
                err = curr_sample_weights[(stump_pred != y)].sum()  # misclassified value's sample weight sum
                stump_weight = np.log((1 - err) / err) / 2  # alpha minimizing misclassified sample's w
            
                X_single = X[:, lowest_error[0]].reshape((-1, 1))
                stump = stump.fit(X_single, y, sample_weight = curr_sample_weights)
                self.hj.append(lowest_error[0])


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

        # previous cycle 까지의 emsemble classifier 로 예측한 값을 저장.
        if t == 0 :  # 첫번째 iteration 인 경우.
            self.pre_e = stump_pred
            self.pre_st_w = stump_weight
        else:
            self.pre_e = np.column_stack([self.pre_e, stump_pred])
            self.pre_st_w = np.vstack([self.pre_st_w, stump_weight])
    
        # t iteration 까지의 emsemble 예측값
        self.pre_y = np.sign(np.dot(self.pre_e, self.pre_st_w)).reshape((X.shape[0],))

    return self


def predict(self, X):
    
    result = []
    for j, sw in zip(self.hj, self.stumps):
        X_single = X[:, j].reshape((-1, 1))
        result.append(sw.predict(X_single))
    result_ = np.array(result).T

    pred_y = np.sign(np.dot(result_, self.stump_weights.reshape((-1,1)))).reshape((X.shape[0],))
    
    return pred_y

##############################################################################



perf = []
for e in tqdm.tqdm(np.arange(0.1,1.01,step=0.1)):
    basket = []
    for rs in [1,2,3,4,5]:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            shuffle=False,
                                                            random_state=rs)
        AdaBoost.fit = fit
        AdaBoost.predict = predict
    
        clf = AdaBoost().fit(X_train, y_train, iters = 1000, eps = e)
    
        test_err = (clf.predict(X_test) != y_test).mean()
        basket.append(test_err)
    
    perf.append(np.mean(basket))
    
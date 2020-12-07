#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 03:25:56 2020

@author: yejin
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
import random
import tqdm 
import math
from sklearn.datasets import make_gaussian_quantiles


##############################################################################

# 1. 데이터 불러오기.

diabetes = pd.read_csv('https://drive.google.com/uc?export=download&id=1Mh72sYe5DejUBqbKZKGcHo_HNts6bfDA', header=0)
liver = pd.read_csv('https://drive.google.com/uc?export=download&id=1DRr2zSx8xsnxeBVTr7M8-43kw71l4nE9', header=0)
ionosphere = pd.read_csv('https://drive.google.com/uc?export=download&id=1_u2ABH8xmVB811vVKWQndSNm0DJLFHVb',header=0)


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
liver = liver[liver['A/G Ratio Albumin and Globulin Ratio'].apply(lambda x: not math.isnan(x))]

# ionosphere dataset
new_class = []
for i in range(len(ionosphere)):
    if ionosphere.loc[i]['label'] == 'g':
        new_class.append(1)
    else:
        new_class.append(-1)
ionosphere['label'] = new_class


#%%
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

def fit_ada(self, X: np.ndarray, y: np.ndarray, iters: int):
    """ Fit the model using training data """

    X, y = self._check_X_y(X, y)
    n = X.shape[0]

    # init numpy arrays
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)

    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n

    for t in range(iters):
        # fit  weak learner
        curr_sample_weights = self.sample_weights[t]
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump = stump.fit(X, y, sample_weight=curr_sample_weights)

        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)
        err = curr_sample_weights[(stump_pred != y)].sum()# / n
        stump_weight = np.log((1 - err) / err) / 2

        # update sample weights
        new_sample_weights = (
            curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
        )
        
        new_sample_weights /= new_sample_weights.sum()

        # If not final iteration, update sample weights for t+1
        if t+1 < iters:
            self.sample_weights[t+1] = new_sample_weights

        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

    return self

def fit_rada(self, X: np.ndarray, y: np.ndarray, iters: int,r_parameter,age_parameter):
    """ Fit the model using training data """

    X, y = self._check_X_y(X, y)
    n = X.shape[0]

    # init numpy arrays
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)
    self.age=np.zeros(len(X))
    self.beta=np.ones(len(X))
    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n

    for t in range(iters):
        # fit  weak learner
        curr_sample_weights = self.sample_weights[t]
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump = stump.fit(X, y, sample_weight=curr_sample_weights)

        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)
        err = curr_sample_weights[(stump_pred != y)].sum()# / n
        
        # update age & beta
        misclf=[int(x) for x in (stump_pred!=y)] # correct =0 / miss = 1
        misclf2=np.array([-x if x==1 else 1 for x in misclf]) # correct = 1 / miss=-1
        self.age+=misclf
        for m in range(len(self.age)):
            if misclf[m] ==0:
                self.beta[m]=1

        for k in range(len(self.age)):
            if self.age[k] > age_parameter:
                self.age[k]=0
                self.beta[k]=-1
        
        if err < r_parameter:
            stump_weight=0.5*(np.log((1-err)/err)**(1/r_parameter))
            +0.5*(np.log((1-r_parameter)/r_parameter))
            -0.5*((np.log((1-r_parameter)/r_parameter))**(1/r_parameter))
        else:
            stump_weight = np.log((1 - err) / err) / 2
        

        # update sample weights
        new_sample_weights = (
            curr_sample_weights * np.exp(-stump_weight * y * stump_pred*self.beta)
        )
        
        new_sample_weights /= new_sample_weights.sum()

        # If not final iteration, update sample weights for t+1
        if t+1 < iters:
            self.sample_weights[t+1] = new_sample_weights

        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

    return self

def fit_dada(self, X: np.ndarray, y: np.ndarray, iters: int, eps=0.03):

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
                X_single = X[:, i].reshape((-1,1))
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


# len(X)
# int(X.size/len(X))
# X3 = X[0:len(X),1]
# X3 = X3.reshape(-1,1)
# X2 = X[1,:].reshape(-1,1).T
# X.shape[0]


def predict(self, X):
    """ Make predictions using already fitted model """
    stump_preds = np.array([stump.predict(X) for stump in self.stumps])
    return np.sign(np.dot(self.stump_weights, stump_preds))

def predict_dada(self, X):
    
    result = []
    for j, sw in zip(self.hj, self.stumps):
        X_single = X[:, j].reshape((-1, 1))
        result.append(sw.predict(X_single))
    result_ = np.array(result).T

    pred_y = np.sign(np.dot(result_, self.stump_weights.reshape((-1,1)))).reshape((X.shape[0],))
    
    return pred_y

def make_toy_dataset(n: int = 100, random_seed: int = None,noise=None):
    """ Generate a toy dataset for evaluating AdaBoost classifiers """
    
    n_per_class = int(n/2)
    
    if random_seed:
        np.random.seed(random_seed)

    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)
    y=y*2-1
    if noise is not None:
        index=np.random.choice(np.arange(len(y)),round(len(y)*noise))
        y[index]=-y[index]
        
    return X, y

def plot_adaboost(X: np.ndarray,
                  y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  annotate: bool = False,
                  ax: Optional[mpl.axes.Axes] = None,
                  dada=True) -> None:
    """ Plot ± samples in 2D, optionally with decision boundary """

    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    if sample_weights is not None:
        sizes = np.array(sample_weights) * X.shape[0] * 100
    else:
        sizes = np.ones(shape=X.shape[0]) * 100

    X_pos = X[y == 1]
    sizes_pos = sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    X_neg = X[y == -1]
    sizes_neg = sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    if clf:
        plot_step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # If all predictions are positive class, adjust color map acordingly
        if list(np.unique(Z)) == [1]:
            fill_colors = ['r']
        else:
            fill_colors = ['b', 'r']

        ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

    if annotate:
        for i, (x, y) in enumerate(X):
            offset = 0.05
            ax.annotate(f'$x_{i + 1}$', (x + offset, y - offset))

    ax.set_xlim(x_min+0.5, x_max-0.5)
    ax.set_ylim(y_min+0.5, y_max-0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def truncate_adaboost(clf, t: int):
    """ Truncate a fitted AdaBoost up to (and including) a particular iteration """
    assert t > 0, 't must be a positive integer'
    from copy import deepcopy
    new_clf = deepcopy(clf)
    new_clf.stumps = clf.stumps[:t]
    new_clf.stump_weights = clf.stump_weights[:t]
    return new_clf    
    
def plot_staged_adaboost(X, y, clf, iters=10):
    """ Plot weak learner and cumulaive strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters*3),
                             nrows=iters,
                             ncols=2,
                             sharex=True,
                             dpi=100)
    
    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}')
        plot_adaboost(X, y, clf.stumps[i],
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax1)

        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}')
        plot_adaboost(X, y, trunc_clf,
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
def noise_data(train_x_data,train_y_data,noise_percent,feature):
    # sampling
    np.random.seed(seed=100)
    index=np.random.choice(len(train_x_data),round(len(train_x_data)*noise_percent))
    # filp label
    noise_data_x=train_x_data[index].copy()
    noise_data_y=train_y_data[index].copy()
    noise_data_y=-noise_data_y
    # add gaussain noise
    noise_data_x[:,feature]=noise_data_x[:,feature]+np.random.normal(0,0.1,(len(noise_data_x),len(feature)))
    
    return np.concatenate([train_x_data,noise_data_x],axis=0),np.concatenate([train_y_data,noise_data_y],axis=0),noise_data_x,noise_data_y

        
##############################################################################
#%%
## original data
# diabetes
#X = np.array(diabetes.iloc[:, :-1])
#y = np.array(diabetes.iloc[:, -1])

# # liver
X = np.array(liver.iloc[:, :-1])
y = np.array(liver.iloc[:, -1])

# # ionosophere
#X = np.array(ionosphere.iloc[:, :-1])
#y = np.array(ionosphere.iloc[:, -1])

AdaBoost.fit_ada = fit_ada
AdaBoost.fit_rada=fit_rada
AdaBoost.fit_dada=fit_dada
AdaBoost.predict = predict
AdaBoost.predict_d=predict_dada


ada_train_err_l=[]
ada_test_err_l=[]
rada_train_err_l=[]
rada_test_err_l=[]
dada_train_err_l=[]
dada_test_err_l=[]

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=i)
    
    clf_ada = AdaBoost().fit_ada(X_train, y_train, iters=100)
    clf_rada = AdaBoost().fit_rada(X_train, y_train, iters=100,r_parameter=0.05,age_parameter=4)    
    clf_dada = AdaBoost().fit_dada(X_train, y_train, iters = 100,eps=0.08)
    
    train_err_ada = (clf_ada.predict(X_train) != y_train).mean()
    train_err_rada = (clf_rada.predict(X_train) != y_train).mean()    
    train_err_dada = (clf_dada.predict_d(X_train) != y_train).mean()
    test_err_ada = (clf_ada.predict(X_test) != y_test).mean()
    test_err_rada = (clf_rada.predict(X_test) != y_test).mean()    
    test_err_dada = (clf_dada.predict_d(X_test) != y_test).mean()
    ada_train_err_l.append(train_err_ada)
    ada_test_err_l.append(test_err_ada)
    rada_train_err_l.append(train_err_rada)
    rada_test_err_l.append(test_err_rada)
    dada_train_err_l.append(train_err_dada)
    dada_test_err_l.append(test_err_dada)
    
print('============== real data ===============')
print('train ada mean :',np.mean(np.array(ada_train_err_l)))
print('train ada std :',np.std(np.array(ada_train_err_l)))
print(' ')
print('test ada mean :',np.mean(np.array(ada_test_err_l)))
print('test ada std :',np.std(np.array(ada_test_err_l)))
print('---------------------------------------')
print('train rada mean :',np.mean(np.array(rada_train_err_l)))
print('train rada std :',np.std(np.array(rada_train_err_l)))
print(' ')
print('test rada mean :',np.mean(np.array(rada_test_err_l)))
print('test rada std :',np.std(np.array(rada_test_err_l)))
print('---------------------------------------')
print('train dada mean :',np.mean(np.array(dada_train_err_l)))
print('train dada std :',np.std(np.array(dada_train_err_l)))
print(' ')
print('test dada mean :',np.mean(np.array(dada_test_err_l)))
print('test dada std :',np.std(np.array(dada_test_err_l)))

#%%
# noisy data
X = np.array(diabetes.iloc[:, :-1])
y = np.array(diabetes.iloc[:, -1])

# # liver
#X = np.array(liver.iloc[:, :-1])
#y = np.array(liver.iloc[:, -1])

# # ionosophere
#X = np.array(ionosphere.iloc[:, :-1])
#y = np.array(ionosphere.iloc[:, -1])

AdaBoost.fit_ada = fit_ada
AdaBoost.fit_rada=fit_rada
AdaBoost.fit_dada=fit_dada
AdaBoost.predict = predict
AdaBoost.predict_d=predict_dada


ada_train_err_l=[]
ada_test_err_l=[]
rada_train_err_l=[]
rada_test_err_l=[]
dada_train_err_l=[]
dada_test_err_l=[]

noise_per=0.3
#[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=i)
    noise_x_train,noise_y_train,noise_x,noise_y=noise_data(X_train,y_train,noise_per,[0,1,2,3,4,5,6,7])
    X_train=np.array(noise_x_train)
    y_train=np.array(noise_y_train)
    
    
    clf_ada = AdaBoost().fit_ada(X_train, y_train, iters=35)
    clf_rada = AdaBoost().fit_rada(X_train, y_train, iters=35,r_parameter=0.4,age_parameter=9)    
    clf_dada = AdaBoost().fit_dada(X_train, y_train, iters=35,eps=0.01)
    
    train_err_ada = (clf_ada.predict(X_train) != y_train).mean()
    train_err_rada = (clf_rada.predict(X_train) != y_train).mean()    
    train_err_dada = (clf_dada.predict_d(X_train) != y_train).mean()
    test_err_ada = (clf_ada.predict(X_test) != y_test).mean()
    test_err_rada = (clf_rada.predict(X_test) != y_test).mean()    
    test_err_dada = (clf_dada.predict_d(X_test) != y_test).mean()
    
    ada_train_err_l.append(train_err_ada)
    ada_test_err_l.append(test_err_ada)
    rada_train_err_l.append(train_err_rada)
    rada_test_err_l.append(test_err_rada)
    dada_train_err_l.append(train_err_dada)
    dada_test_err_l.append(test_err_dada)
    
print('============= noisy data {0} ================='.format(noise_per))
print('train ada mean :',np.mean(np.array(ada_train_err_l)))
print('train ada std :',np.std(np.array(ada_train_err_l)))
print(' ')
print('test ada mean :',np.mean(np.array(ada_test_err_l)))
print('test ada std :',np.std(np.array(ada_test_err_l)))
print('---------------------------------------')
print('train rada mean :',np.mean(np.array(rada_train_err_l)))
print('train rada std :',np.std(np.array(rada_train_err_l)))
print(' ')
print('test rada mean :',np.mean(np.array(rada_test_err_l)))
print('test rada std :',np.std(np.array(rada_test_err_l)))
print('---------------------------------------')
print('train dada mean :',np.mean(np.array(dada_train_err_l)))
print('train dada std :',np.std(np.array(dada_train_err_l)))
print(' ')
print('test dada mean :',np.mean(np.array(dada_test_err_l)))
print('test dada std :',np.std(np.array(dada_test_err_l)))

#%%
# weight of noise data plot
weight_l_ada=[]
weight_l_rada=[]
weight_l_dada=[]
iters=35
for i in range(clf_ada.sample_weights.shape[0]):    
    weight_l_ada.append(np.mean(clf_ada.sample_weights[i][:len(noise_x)]))
for i in range(clf_rada.sample_weights.shape[0]):    
    weight_l_rada.append(np.mean(clf_rada.sample_weights[i][:len(noise_x)]))
for i in range(clf_dada.sample_weights.shape[0]):    
    weight_l_dada.append(np.mean(clf_dada.sample_weights[i][:len(noise_x)]))

plt.plot(np.arange(iters),np.array(weight_l_ada)*100)
plt.plot(np.arange(iters),np.array(weight_l_rada)*100)
plt.plot(np.arange(iters),np.array(weight_l_dada)*100)
plt.ylabel('mean of normal data weight')
plt.xlabel('iteration')
plt.legend(['adaboost','radaboost','dadaboost'])
plt.show()
#%%
#plot staged adaboost (ada,rada)
toy_x,toy_y=make_toy_dataset(n=20,random_seed=4,noise=0.4)
clf_rada = AdaBoost().fit_rada(toy_x,toy_y,iters=10,r_parameter=0.45,age_parameter=8)#,r_parameter=0.45,age_parameter=5
plot_staged_adaboost(toy_x,toy_y, clf_rada)
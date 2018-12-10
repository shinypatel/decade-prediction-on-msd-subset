#!/bin/python3

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier as ada
import math
import numpy as np
import time


''' other useful resources:
    
    https://scikit-learn.org/stable/modules/svm.html
'''


def split(X, y, test_size=0.25, random_state=None):
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(
        X, y,
        test_size=test_size,
        # None defaults to RandomState instance used by np.random
        random_state=random_state,
        stratify=True
    )


def train_and_test(X_train, y_train, X_test, y_test):
    ''' https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
        https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV '''
    C, max_iter = [math.pow(2, i) for i in range(-4, 8)], [1e6]
    #param_grid = [{}]
    #param_grid = [{'loss': ['squared_hinge'], 'dual': [False],
                   #'C': C, 'max_iter': max_iter},
                  #{'loss': ['hinge'], 'C': C, 'max_iter': max_iter}]
    param_grid = [{"n_estimators": [10]}, {"n_estimators": [50]},{"n_estimators": [100]},
            {"n_estimators": [300]},{"n_estimators": [500]},{"n_estimators": [1000]}]
    #param_grid = [{"n_estimators": [10,50,100,300,500,1000]}]

    ''' https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
        
        implements "one-vs-the-rest" multi-class strategy
        (preferred to "one-vs-one" because of significantly less runtime for
        similar results '''
    clf = GridSearchCV(ada(n_estimators=500),
                       param_grid,
                       scoring='accuracy',
                       iid=False,  # return average score across folds
                       cv=3)

    clf.fit(X_train, y_train)
    print('Best params set found on training set:\n', clf.best_params_)

    print('\nGrid (mean accuracy) scores on training set:\n')
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))

    print('\nDetailed classification report:\n')
    y_pred = clf.predict(X_test)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    print(classification_report(y_test, y_pred))
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


def decade(year):
    return int(math.floor(year / 10) * 10)


if __name__ == '__main__':
    start_time = time.time()

    #data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=514846)
    data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=450000)
    X = StandardScaler().fit_transform(data[:, 1:])
    y = np.vectorize(decade)(data[:, 0])
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_and_test(X_train, y_train, X_test, y_test)

    print('\nRunning time: %d min' % int((time.time() - start_time) / 60))

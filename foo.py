#!/bin/python3

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.metrics import accuracy_score
import math


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


def feature_selector(X_train, y_train, estimator, k=3):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
        https://stackoverflow.com/questions/45925011/feature-selection-with-cross-validation-using-scikit-learn-for-anova-test '''
    selector = RFECV(
        estimator,
        cv=k,
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scoring='accuracy'
    )
    selector = selector.fit(X_train, y_train)
    return selector.support_  # mask of selected features


def svm_clf(C=1.0):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

        implements "one-vs-the-rest" multi-class strategy
        (preferred to "one-vs-one" because of significantly less runtime for
        similar results '''
    clf = svm.LinearSVC(
        dual=False,  # preferred when n_samples > n_features
        C=C
    )
    return clf


def param_tuning(X_train, y_train, estimator, param_grid, k=3):
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring='accuracy',
        iid=False,  # return average score across folds
        cv=k,
        # computationally expensive and not required to select best parameters
        return_train_score=False
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def svm_predictions(X_train, y_train, X_test):
    ''' https://www.hackerearth.com/blog/machine-learning/simple-tutorial-svm-parameter-tuning-python-r/
        https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
        https://stats.stackexchange.com/questions/43943/which-search-range-for-determining-svm-optimal-c-and-gamma-parameters '''
    param_grid = {'C': [math.pow(2, i) for i in range(-4, 8)]}
    best_params = param_tuning(X_train, y_train, svm_clf(), param_grid)

    clf = svm_clf(best_params['C'])
    clf.fit(X_train, y_train)
    return clf.predict(X_test)   # predicted class labels


def evaluate(d, y_test):
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics
    print('Accuracy: %.2f' % accuracy_score(d, y_test))


if __name__ == '__main__':
    # dummy val
    from sklearn.datasets import make_classification
    X, y = make_classification(n_features=4, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    best_features = feature_selector(X_train, y_train, svm_clf())
    X_train = X_train[:, best_features]
    X_test = X_test[:, best_features]

    evaluate(svm_predictions(X_train, y_train, X_test), y_test)

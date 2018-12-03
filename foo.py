#!/bin/python3

from sklearn.feature_selection import RFECV
from sklearn import svm


def feature_selector(estimator, X, Y, k=3):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
        https://stackoverflow.com/questions/45925011/feature-selection-with-cross-validation-using-scikit-learn-for-anova-test '''
    selector = RFECV(
        estimator,
        cv=k,  # k-fold cross validation
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scoring='accuracy'
    )
    selector = selector.fit(X, Y)
    return selector.support_  # mask of selected features


def svm_clf():
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

        implements "one-vs-the-rest" multi-class strategy
        (preferred to "one-vs-one" because of significantly less runtime for
        similar results '''
    clf = svm.LinearSVC(
        dual=False,  # preferred when n_samples > n_features
    )
    return clf


# def svm_param_tuning():


if __name__ == '__main__':
    X, Y = None, None
    X = [feature_selector(svm_clf(), X, Y)]

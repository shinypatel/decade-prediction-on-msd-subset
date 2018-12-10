#!/bin/python3
# Shahriar Hooshmand,
# CSE5523 final project,
# Neural Network Optimization,
# Dec2018, Ohio State University

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import cm

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
        # TODO: test w/ hinge loss
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
    #print('Accuracy: %.2f' % accuracy_score(d, y_test))
    return accuracy_score(d, y_test) *100

def round_down(num):
    return num - (num%10)

def tup (n_layers, n_neur):
    dum =()
    for i in range(n_layers):
        dum += (n_neur,)
    return dum

# TODO: svm plot


if __name__ == '__main__':
    # dummy val
    from sklearn.datasets import make_classification
##############
# This section does the contour plot analysis
    total = 515345
    data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=514846)

    #data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=257000)
    Y = data[:, 0]
    X = data[:, 1:]

    for i in range(0,len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)

    X, y = StandardScaler().fit_transform(X), Y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    acc = []
    plt.figure()
    lin = ['--', '-', ':']
    cycol = cycle('bgrkmyc')
    acc =[]

    for i in range(1,40):
        for j in range(1,40):
            mlp = MLPClassifier(random_state=5,hidden_layer_sizes=tup(i,j), max_iter=1000)
            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            ev = evaluate(predictions, y_test)
            print(i,j,ev)
            acc.append ((i,j,ev))
        # plt.plot(range(1, 40), acc, lab = "#layers" + str(i )  , ls= lin[i], title="Ti-0.498 at pct O : T = 423K" )
        # plt.scatter(range(1, 40), acc, ls= lin[i])
    A = np.array(acc)
    np.save("mesh_file",A)

#########################################################################################################

# This section goes to generating the contourplot

    data = np.load("mesh_file.npy")


    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.view_init(azim=0, elev=-90)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 39, 100 ), np.linspace(0, 39, 100 ))

    grid_z = griddata(data[:,0:2], data[:,2], (grid_x, grid_y), method='nearest')
    z_min, z_max = grid_z.min(), grid_z.max()

    plt.figure()
    cp = plt.contourf(grid_x, grid_y,grid_z,cmap="jet")
    # v = np.linspace(50, 90, 15, endpoint=True)
    cbar= plt.colorbar(cp)
    cbar.ax.set_ylabel('Accuracy %', rotation=90)
    plt.title('Layers/Neuron Tuning ')
    plt.xlabel('# Layers')
    plt.ylabel('# Neurons')
    plt.savefig("neuron_layers_contour.pdf")
#################################################################################################################
# This section goes to the test analysis on hyperparameters
    total = 515345
    data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=514846)
    #data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=257000)

    Y = data[:, 0]
    X = data[:, 1:]

    for i in range(0,len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)



    X, y = StandardScaler().fit_transform(X), Y

    # for i in range(np.shape(X)[0] ):
    #     for j in range(np.shape(X)[1] ):
    #         if (X[i][j]) <0:
    #             X[i][j] = X[i][j]* -1
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # best_features = SelectKBest(chi2, k=20)
    # best_features = SelectPercentile(chi2, percentile=80)
    # X_train = best_features.fit_transform(X_train,y_train)
    # X_test = best_features.fit_transform(X_test, y_test)


    eta= list(np.arange(0.05,0.5,0.05))
    alp = [0, 0.001, 0.01, 0.5, 0.9]

    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    lin = ['--', '-', ':']
    cycol = cycle('bgrkmyc')

    for al in alp:
        acc =[]
        for et in eta:
            cl=next(cycol)
            # mlp = MLPClassifier( random_state=5, hidden_layer_sizes=tup(2,38), alpha=al,learning_rate_init = et, tol=1e-4)
            mlp = MLPClassifier( random_state=5, hidden_layer_sizes=tup(2,39), alpha=al,learning_rate_init = et, max_iter=10000)

            mlp.fit(X_train,y_train)
            predictions = mlp.predict(X_test)
            evaluate(predictions, y_test)
            acc.append(evaluate(predictions, y_test))
            #acc.append(mlp.n_iter_)
        plt.plot(eta, acc, label = r"$\alpha = $" + str( al ) , color=cl )
        plt.scatter(eta, acc, color= cl)

    plt.xlabel(r"$\eta$")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.legend()
    plt.savefig("layer_test2.pdf")
    #################################################################################################################
# This part goes to the model predictions on actual dataset after parameter optimization

    data = np.loadtxt('YearPredictionMSD.txt', delimiter=',', skiprows=257000)

    Y = data[:, 0]
    X = data[:, 1:]

    for i in range(0, len(Y)):
        Y[i] = round_down(np.abs(Y[i]) % 100)

    X, y = StandardScaler().fit_transform(X), Y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    eta = list(np.arange(0.05, 0.5, 0.05))
    alp = [0, 0.001, 0.01, 0.5, 0.9]

    eta = [0.1]
    alp = [0.5]

    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    lin = ['--', '-', ':']
    cycol = cycle('bgrkmyc')

    for al in alp:
        acc = []
        for et in eta:
            cl = next(cycol)
            mlp = MLPClassifier(random_state=5, hidden_layer_sizes=tup(2, 39), alpha=al, learning_rate_init=et,
                                max_iter=10000)

            mlp.fit(X_train, y_train)
            predictions = mlp.predict(X_test)
            evaluate(predictions, y_test)
            acc.append(evaluate(predictions, y_test))
            # acc.append(mlp.n_iter_)

    #################################################################################################################

print("done")

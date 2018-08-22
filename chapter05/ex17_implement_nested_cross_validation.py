import numpy as np
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC


def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    for training_samples, test_samples in outer_cv.split(X, y):
        best_params = {}
        best_score = -1
        for params in parameter_grid:
            cv_scores = []
            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
                clf = Classifier(**params)
                clf.fit(X[inner_train], y[inner_train])
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        outer_score = clf.score(X[test_samples], y[test_samples])
        outer_scores.append(outer_score)
    return np.array(outer_scores)

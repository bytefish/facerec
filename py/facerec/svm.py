#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from facerec.classifier import SVM
from facerec.util import asRowMatrix


def grid_search(model, X, y, tuned_parameters):
    # Check if the Classifier in the Model is actually an SVM:
    if not isinstance(model.classifier, SVM):
        raise TypeError("classifier must be of type SVM!")
    # First compute the features for this SVM-based model:
    features = model.feature.compute(X,y)
    # Turn the List of Features into a matrix with each feature as Row:
    Xrow = asRowMatrix(features)
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(Xrow, y, test_size=0.5, random_state=0)
    # Define the Classifier:
    scores = ['precision', 'recall']
    # Evaluate the Model:
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                        scoring='%s_macro' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
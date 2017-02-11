#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import logging
import math as math
import random as random
from builtins import range

import numpy as np

from facerec.model import PredictableModel


# TODO The evaluation of a model should be completely moved to the generic ValidationStrategy. The specific Validation 
#       implementations should only care about partition the data, which would make a lot sense. Currently it is not 
#       possible to calculate the true_negatives and false_negatives with the way the predicitions are generated and 
#       data is prepared.
#       
#     The mentioned problem makes a change in the PredictionResult necessary, which basically means refactoring the 
#       entire framework. The refactoring is planned, but I can't give any details as time of writing.
#
#     Please be careful, when referring to the Performance Metrics at the moment, only the Precision is implemented,
#       and the rest has no use at the moment. Due to the missing refactoring discussed above.
#

def shuffle(X, y):
    """ Shuffles two arrays by column (len(X) == len(y))
        
        Args:
        
            X [dim x num_data] input data
            y [1 x num_data] classes

        Returns:

            Shuffled input arrays.
    """
    idx = np.argsort([random.random() for i in range(len(y))])
    y = np.asarray(y)
    X = [X[i] for i in idx]
    y = y[idx]
    return (X, y)
    
def slice_2d(X,rows,cols):
    """
    
    Slices a 2D list to a flat array. If you know a better approach, please correct this.
    
    Args:
    
        X [num_rows x num_cols] multi-dimensional data
        rows [list] rows to slice
        cols [list] cols to slice
    
    Example:
    
        >>> X=[[1,2,3,4],[5,6,7,8]]
        >>> # slice first two rows and first column
        >>> Commons.slice(X, range(0,2), range(0,1)) # returns [1, 5]
        >>> Commons.slice(X, range(0,1), range(0,4)) # returns [1,2,3,4]
    """
    return [X[i][j] for j in cols for i in rows]

def precision(true_positives, false_positives):
    """Returns the precision, calculated as:
        
        true_positives/(true_positives+false_positives)
        
    """
    return accuracy(true_positives, 0, false_positives, 0)
    
def accuracy(true_positives, true_negatives, false_positives, false_negatives, description=None):
    """Returns the accuracy, calculated as:
    
        (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)
        
    """
    true_positives = float(true_positives)
    true_negatives = float(true_negatives)
    false_positives = float(false_positives)
    false_negatives = float(false_negatives)
    if (true_positives + true_negatives + false_positives + false_negatives) < 1e-15:
       return 0.0
    return (true_positives+true_negatives)/(true_positives+false_positives+true_negatives+false_negatives)

class ValidationResult(object):
    """Holds a validation result.
    """
    def __init__(self, true_positives, true_negatives, false_positives, false_negatives, description):
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.description = description
        
    def __repr__(self):
        res_precision = precision(self.true_positives, self.false_positives) * 100
        res_accuracy = accuracy(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives) * 100
        return "ValidationResult (Description=%s, Precision=%.2f%%, Accuracy=%.2f%%)" % (self.description, res_precision, res_accuracy)
    
class ValidationStrategy(object):
    """ Represents a generic Validation kernel for all Validation strategies.
    """
    def __init__(self, model):
        """    
        Initialize validation with empty results.
        
        Args:
        
            model [PredictableModel] The model, which is going to be validated.
        """
        if not isinstance(model,PredictableModel):
            raise TypeError("Validation can only validate the type PredictableModel.")
        self.model = model
        self.validation_results = []
    
    def add(self, validation_result):
        self.validation_results.append(validation_result)
        
    def validate(self, X, y, description):
        """
        
        Args:
            X [list] Input Images
            y [y] Class Labels
            description [string] experiment description
        
        """
        raise NotImplementedError("Every Validation module must implement the validate method!")
        
    
    def print_results(self):
        print(self.model)
        for validation_result in self.validation_results:
            print(validation_result)

    def __repr__(self):
        return "Validation Kernel (model=%s)" % (self.model)
        
class KFoldCrossValidation(ValidationStrategy):
    """ 
    
    Divides the Data into 10 equally spaced and non-overlapping folds for training and testing.
    
    Here is a 3-fold cross validation example for 9 observations and 3 classes, so each observation is given by its index [c_i][o_i]:
                
        o0 o1 o2        o0 o1 o2        o0 o1 o2  
    c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |
    c1 | A  B  B |  c1 | B  A  B |  c1 | B  B  A |
    c2 | A  B  B |  c2 | B  A  B |  c2 | B  B  A |
    
    Please note: If there are less than k observations in a class, k is set to the minimum of observations available through all classes.
    """
    def __init__(self, model, k=10):
        """
        Args:
            k [int] number of folds in this k-fold cross-validation (default 10)
        """
        super(KFoldCrossValidation, self).__init__(model=model)
        self.k = k
        self.logger = logging.getLogger("facerec.validation.KFoldCrossValidation")

    def validate(self, X, y, description="ExperimentName"):
        """ Performs a k-fold cross validation
        
        Args:

            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        X,y = shuffle(X,y)
        c = len(np.unique(y))
        foldIndices = []
        n = np.iinfo(np.int).max
        for i in range(0,c):
            idx = np.where(y==i)[0]
            n = min(n, idx.shape[0])
            foldIndices.append(idx.tolist()); 

        # I assume all folds to be of equal length, so the minimum
        # number of samples in a class is responsible for the number
        # of folds. This is probably not desired. Please adjust for
        # your use case.
        if n < self.k:
            self.k = n

        foldSize = int(math.floor(n/self.k))
        
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        for i in range(0,self.k):
        
            self.logger.info("Processing fold %d/%d." % (i+1, self.k))
                
            # calculate indices
            l = int(i*foldSize)
            h = int((i+1)*foldSize)
            testIdx = slice_2d(foldIndices, cols=range(l,h), rows=range(0, c))
            trainIdx = slice_2d(foldIndices,cols=range(0,l), rows=range(0,c))
            trainIdx.extend(slice_2d(foldIndices,cols=range(h,n),rows=range(0,c)))
            
            # build training data subset
            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
                        
            self.model.compute(Xtrain, ytrain)
            
            # TODO I have to add the true_negatives and false_negatives. Models also need to support it,
            # so we should use a PredictionResult, instead of trying to do this by simply comparing
            # the predicted and actual class.
            #
            # This is inteneded of the next version! Feel free to contribute.
            for j in testIdx:
                prediction = self.model.predict(X[j])[0]
                if prediction == y[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
                    
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "k-Fold Cross Validation (model=%s, k=%s)" % (self.model, self.k)

class LeaveOneOutCrossValidation(ValidationStrategy):
    """ Leave-One-Cross Validation (LOOCV) uses one observation for testing and the rest for training a classifier:

        o0 o1 o2        o0 o1 o2        o0 o1 o2           o0 o1 o2
    c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |     c0 | B  B  B |
    c1 | B  B  B |  c1 | B  B  B |  c1 | B  B  B |     c1 | B  B  B |
    c2 | B  B  B |  c2 | B  B  B |  c2 | B  B  B | ... c2 | B  B  A |
    
    Arguments:
        model [Model] model for this validation
        ... see [Validation]
    """

    def __init__(self, model):
        """ Intialize Cross-Validation module.
        
        Args:
            model [Model] model for this validation
        """
        super(LeaveOneOutCrossValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.LeaveOneOutCrossValidation")
        
    def validate(self, X, y, description="ExperimentName"):
        """ Performs a LOOCV.
        
        Args:
            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        #(X,y) = shuffle(X,y)
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        n = y.shape[0]
        for i in range(0,n):
            
            self.logger.info("Processing fold %d/%d." % (i+1, n))
            
            # create train index list
            trainIdx = []
            trainIdx.extend(range(0,i))
            trainIdx.extend(range(i+1,n))
            
            # build training data/test data subset
            Xtrain = [X[t] for t in trainIdx]
            ytrain = y[trainIdx]
            
            # compute the model
            self.model.compute(Xtrain, ytrain)
            
            # get prediction
            prediction = self.model.predict(X[i])[0]
            if prediction == y[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
                
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "Leave-One-Out Cross Validation (model=%s)" % (self.model)

class LeaveOneClassOutCrossValidation(ValidationStrategy):
    """ Leave-One-Cross Validation (LOOCV) uses one observation for testing and the rest for training a classifier:

        o0 o1 o2        o0 o1 o2        o0 o1 o2           o0 o1 o2
    c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |     c0 | B  B  B |
    c1 | B  B  B |  c1 | B  B  B |  c1 | B  B  B |     c1 | B  B  B |
    c2 | B  B  B |  c2 | B  B  B |  c2 | B  B  B | ... c2 | B  B  A |
    
    Arguments:
        model [Model] model for this validation
        ... see [Validation]
    """

    def __init__(self, model):
        """ Intialize Cross-Validation module.
        
        Args:
            model [Model] model for this validation
        """
        super(LeaveOneClassOutCrossValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.LeaveOneClassOutCrossValidation")
        
    def validate(self, X, y, g, description="ExperimentName"):
        """
        TODO Add example and refactor into proper interface declaration.
        """
        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        
        for i in range(0,len(np.unique(y))):
            self.logger.info("Validating Class %s." % i)
            # create folds
            trainIdx = np.where(y!=i)[0]
            testIdx = np.where(y==i)[0]
            # build training data/test data subset
            Xtrain = [X[t] for t in trainIdx]
            gtrain = g[trainIdx]
            
            # Compute the model, this time on the group:
            self.model.compute(Xtrain, gtrain)
            
            for j in testIdx:
                # get prediction
                prediction = self.model.predict(X[j])[0]
                if prediction == g[j]:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))
    
    def __repr__(self):
        return "Leave-One-Class-Out Cross Validation (model=%s)" % (self.model)

class SimpleValidation(ValidationStrategy):
    """Implements a simple Validation, which allows you to partition the data yourself.
    """
    def __init__(self, model):
        """
        Args:
            model [PredictableModel] model to perform the validation on
        """
        super(SimpleValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.SimpleValidation")

    def validate(self, Xtrain, ytrain, Xtest, ytest, description="ExperimentName"):
        """

        Performs a validation given training data and test data. User is responsible for non-overlapping assignment of indices.

        Args:
            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        self.logger.info("Simple Validation.")
       
        self.model.compute(Xtrain, ytrain)

        self.logger.debug("Model computed.")

        true_positives, false_positives, true_negatives, false_negatives = (0,0,0,0)
        count = 0
        for i in ytest:
            self.logger.debug("Predicting %s/%s." % (count, len(ytest)))
            prediction = self.model.predict(Xtest[i])[0]
            if prediction == ytest[i]:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
            count = count + 1
        self.add(ValidationResult(true_positives, true_negatives, false_positives, false_negatives, description))

    def __repr__(self):
        return "Simple Validation (model=%s)" % (self.model)

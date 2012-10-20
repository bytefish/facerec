import numpy as np
import math as math
import random as random
import logging

from facerec.model import PredictableModel
from facerec.classifier import AbstractClassifier

def shuffle(X, y):
    """ Shuffles two arrays by column (len(X) == len(y))
        
        Args:
        
            X [dim x num_data] input data
            y [1 x num_data] classes

        Returns:

            Shuffled input arrays.
    """
    idx = np.argsort([random.random() for i in xrange(len(y))])
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

class Validation(object):
    """ Represents a generic Validation kernel for all Validation strategies.
    
    Attributes:

        (tp,...,fns)    [int]        Validation results
        accuracy        [float]     Accuracy over all runs.
        std_accuracy    [float]     stddev over all runs.
        runs            [int]         Runs performed for this validation.
        
    """
    def __init__(self, model):
        """    
        Initialize validation with empty results.
        """
        # make sure we can predict on this one
        if not isinstance(model,PredictableModel):
            raise TypeError("Validation can only validate the type PredictableModel.")
        self.model = model
        self._results = np.empty((0,4), np.int)
    
    def add(self, result):
        """ Adds a validation result [tp,fp,tn,fn] to this Validation object.
        
        Args:
        
            result [list] [tp, fp, tn, fn]
        """
        if len(result) != 4:
            return
        self._results = np.vstack((self._results, result))
        
    def at(self, idx, row=None):
        """ 
        
        Returns idx columns of the results list (all if None given).
        
        Args:
        
            idx [int] Column to return (0 == tp, 1 == fp, 2 == tn, 3 == fn)
            row [int] Rows to include
            
        Returns: 
            List with sliced results.
        """
        if row is None:
            row = range(0,self._results.shape[0])
        if idx is None:
            idx = range(0, self._results.shape[1])
        ret = self._results[row, idx]
        if np.size(ret) == 1:
            return np.int(ret)
        return ret
        
    def validate(self, X, y):
        raise NotImplementedError("Every Validation module must implement the validate method!")

    @property
    def tp(self):
        return self.at(0)

    @property
    def fp(self):
        return self.at(1)
    
    @property
    def tn(self):
        return self.at(2)
    
    @property
    def fn(self):
        return self.at(3)
        
    @property
    def tps(self):
        return np.sum(self.at(0))
    
    @property
    def fps(self):
        return np.sum(self.at(1))
    
    @property
    def tns(self):
        return np.sum(self.at(2))

    @property
    def fns(self):
        return np.sum(self.at(3))

    @property
    def runs(self):
        return self._results.shape[0]
    
    @property
    def results(self):
        return self._results

    @property
    def accuracy(self):
        tps = float(self.tps)
        fps = float(self.fps)
        if (tps+fps) < 1e-15:
            return 0.0
        return (tps/(tps+fps))
        
    @property
    def std_accuracy(self):
        try:
            accs = (np.array(self.tp, dtype=np.float32)/(np.array(self.tp+self.fp, dtype=np.float32)))
        except:
            return 0.0
        return np.std(accs)
        
    def __repr__(self):
        return "Validation Kernel (model=%s, runs=%s, accuracy=%.2f%%, std(accuracy)=%.2f%%, tp=%s, fp=%s, tn=%s, fn=%s)" % (self.model, self.runs, (self.accuracy*100), (self.std_accuracy*100), self.tps, self.fps, self.tns, self.fns)
        
class KFoldCrossValidation(Validation):
    """ 
    
    Divides the Data into 10 equally spaced and non-overlapping folds for training and testing.
    
    Here is a 3-fold cross validation example for 9 observations and 3 classes, so each observation is given by its index [c_i][o_i]:
                
        o0 o1 o2        o0 o1 o2        o0 o1 o2  
    c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |
    c1 | A  B  B |  c1 | B  A  B |  c1 | B  B  A |
    c2 | A  B  B |  c2 | B  A  B |  c2 | B  B  A |
    
    Please note: If there are less than k observations in a class, k is set to the minimum of observations available through all classes.
    
    Arguments:
    
        model [Model] model for this validation
        ... see [Validation]
    """
    def __init__(self, model, k=10, results_per_fold=False):
        """
        Args:
            model [Model] model to perform the validation on
            k [int] number of folds in this k-fold cross-validation (default 10)
            results_per_fold [bool] store results per fold (default False)
        """
        super(KFoldCrossValidation, self).__init__(model=model)
        self.k = k
        self.results_per_fold = results_per_fold
        self.logger = logging.getLogger("facerec.validation.KFoldCrossValidation")

    def validate(self, X, y):
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
        
        tp, fp, tn, fn = (0,0,0,0)
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

            for j in testIdx:
                prediction = self.model.predict(X[j])[0]
                if prediction == y[j]:
                    tp = tp + 1
                else:
                    fp = fp + 1

            # add result for this foldIndex
            if self.results_per_fold:
                self.add([tp, fp, tn, fn])
                tp, fp, tn, fn = (0,0,0,0)
                    
        # add k-fold cv results
        if not self.results_per_fold:
            self.add([tp, fp, tn, fn])
    
    def __repr__(self):
        return "k-Fold Cross Validation (model=%s, k=%s, runs=%s, accuracy=%.2f%%, std(accuracy)=%.2f%%, tp=%s, fp=%s, tn=%s, fn=%s)" % (self.model, self.k, self.runs, (self.accuracy*100.0), (self.std_accuracy*100.0), self.tps, self.fps, self.tns, self.fns)

class LeaveOneOutCrossValidation(Validation):
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
        
    def validate(self, X, y):
        """ Performs a LOOCV.
        
        Args:
            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        #(X,y) = shuffle(X,y)
        tp, fp, tn, fn = (0,0,0,0)
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
                tp = tp + 1
            else:
                fp = fp + 1
        self.add([tp, fp, tn, fn])
    
    def __repr__(self):
        return "Leave-One-Out Cross Validation (model=%s, runs=%d, accuracy=%.2f%%, tp=%s, fp=%s, tn=%s, fn=%s)" % (self.model, self.runs, (self.accuracy * 100.0), self.tps, self.fps, self.tns, self.fns)

class LeaveOneClassOutCrossValidation(Validation):
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
        
    def validate(self, X, y, g):
        #(X,y) = shuffle(X,y)
        tp, fp, tn, fn = (0,0,0,0)
        
        for i in range(0,len(np.unique(y))):
            self.logger.info("Validating Class %s." % i)
            # create folds
            trainIdx = np.where(y!=i)[0]
            testIdx = np.where(y==i)[0]
            # build training data/test data subset
            Xtrain = [X[t] for t in trainIdx]
            gtrain = g[trainIdx]
            
            # compute the model (this time on the group!)
            self.model.compute(Xtrain, gtrain)
            
            for j in testIdx:
                # get prediction
                prediction = self.model.predict(X[j])[0]
                if prediction == g[j]:
                    tp = tp + 1
                else:
                    fp = fp + 1
        self.add([tp, fp, tn, fn])
    
    def __repr__(self):
        return "Leave-One-Class-Out Cross Validation (model=%s, runs=%d, accuracy=%.2f%%, tp=%s, fp=%s, tn=%s, fn=%s)" % (self.model, self.runs, (self.accuracy * 100.0), self.tps, self.fps, self.tns, self.fns)

class SimpleValidation(Validation):
    """
    """
    def __init__(self, model):
        """
        Args:
            model [Model] model to perform the validation on
        """
        super(SimpleValidation, self).__init__(model=model)
        self.logger = logging.getLogger("facerec.validation.SimpleValidation")
            
    def validate(self, X, y, trainIdx, testIdx):
        """
        Performs a validation given training data and test data. User is responsible for non-overlapping assignment of indices.

        Args:
            X [dim x num_data] input data to validate on
            y [1 x num_data] classes
        """
        self.logger.info("Simple Validation.")
        # build training data/test from given indices
        Xtrain = [X[t] for t in trainIdx]
        ytrain = y[trainIdx]
        
        # now compute the model
        self.model.compute(Xtrain, ytrain)

        self.logger.debug("Model computed.")

        tp, fp, tn, fn = (0,0,0,0)
        count = 0
        for i in testIdx:
            self.logger.debug("Predicting %s/%s." % (count, len(testIdx)))
            prediction = self.model.predict(X[i])[0]
            if prediction == y[i]:
                tp = tp + 1
            else:
                fp = fp + 1
            count = count + 1
        self.add([tp, fp, tn, fn])
        
    def __repr__(self):
        return "Simple Validation (model=%s, runs=%s, accuracy=%.2f%%, std(accuracy)=%.2f%%, tp=%s, fp=%s, tn=%s, fn=%s)" % (self.model, self.runs, (self.accuracy*100.0), (self.std_accuracy*100.0), self.tps, self.fps, self.tns, self.fns)

from facerec.distance import EuclideanDistance
from facerec.util import asRowMatrix
import logging
import numpy as np
import operator as op

class AbstractClassifier(object):
    def compute(self,X,y):
        raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
    
    def predict(self,X):
        raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

class NearestNeighbor(AbstractClassifier):
    """
    Implements a k-Nearest Neighbor Model with a generic distance metric.
    """
    def __init__(self, dist_metric=EuclideanDistance(), k=1):
        AbstractClassifier.__init__(self)
        self.k = k
        self.dist_metric = dist_metric

    def compute(self, X, y):
        self.X = X
        self.y = np.asarray(y)
    
    def predict(self, q):
        """
        Predicts the k-nearest neighbor for a given query in q. 
        
        Args:
        
            q: The given query sample, which is an array.
            
        Returns:
        
            A list with the classifier output. In this framework it is
            assumed, that the predicted class is always returned as first
            element. Moreover, this class returns the distances for the 
            first k-Nearest Neighbors. 
            
            Example:
            
                [ 0, 
                   { 'labels'    : [ 0,      0,      1      ],
                     'distances' : [ 10.132, 10.341, 13.314 ]
                   }
                ]
            
            So if you want to perform a thresholding operation, you could 
            pick the distances in the second array of the generic classifier
            output.    
                    
        """
        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_metric(xi, q)
            distances.append(d)
        if len(distances) > len(self.y):
            raise Exception("More distances than classes. Is your distance metric correct?")
        distances = np.asarray(distances)
        # Get the indices in an ascending sort order:
        idx = np.argsort(distances)
        # Sort the labels and distances accordingly:
        sorted_y = self.y[idx]
        sorted_distances = distances[idx]
        # Take only the k first items:
        sorted_y = sorted_y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]
        # Make a histogram of them:
        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
        # And get the bin with the maximum frequency:
        predicted_label = max(hist.iteritems(), key=op.itemgetter(1))[0]
        # A classifier should output a list with the label as first item and
        # generic data behind. The k-nearest neighbor classifier outputs the 
        # distance of the k first items. So imagine you have a 1-NN and you
        # want to perform a threshold against it, you should take the first
        # item 
        return [predicted_label, { 'labels' : sorted_y, 'distances' : sorted_distances }]
        
    def __repr__(self):
        return "NearestNeighbor (k=%s, dist_metric=%s)" % (self.k, repr(self.dist_metric))

# libsvm
try:
    from svmutil import *
except ImportError:
    logger = logging.getLogger("facerec.classifier.SVM")
    logger.debug("Import Error: libsvm bindings not available.")
except:
    logger = logging.getLogger("facerec.classifier.SVM")
    logger.debug("Import Error: libsvm bindings not available.")

import sys
from StringIO import StringIO
bkp_stdout=sys.stdout

class SVM(AbstractClassifier):
    """
    This class is just a simple wrapper to use libsvm in the 
    CrossValidation module. If you don't use this framework
    use the validation methods coming with LibSVM, they are
    much easier to access (simply pass the correct class 
    labels in svm_predict and you are done...).

    The grid search method in this class is somewhat similar
    to libsvm grid.py, as it performs a parameter search over
    a logarithmic scale.    Again if you don't use this framework, 
    use the libsvm tools as they are much easier to access.

    Please keep in mind to normalize your input data, as expected
    for the model. There's no way to assume a generic normalization
    step.
    """

    def __init__(self, param=None):
        AbstractClassifier.__init__(self)
        self.logger = logging.getLogger("facerec.classifier.SVM")
        self.param = param
        self.svm = svm_model()
        self.param = param
        if self.param is None:
            self.param = svm_parameter("-q")
    
    def compute(self, X, y):
        self.logger.debug("SVM TRAINING (C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % (self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree))
        # turn data into a row vector (needed for libsvm)
        X = asRowMatrix(X)
        y = np.asarray(y)
        problem = svm_problem(y, X.tolist())        
        self.svm = svm_train(problem, self.param)
        self.y = y
    
    def predict(self, X):
        """
        
        Args:
        
            X: The query image, which is an array.
        
        Returns:
        
            A list with the classifier output. In this framework it is
            assumed, that the predicted class is always returned as first
            element. Moreover, this class returns the libsvm output for
            p_labels, p_acc and p_vals. The libsvm help states:
            
                p_labels: a list of predicted labels
                p_acc: a tuple including  accuracy (for classification), mean-squared 
                   error, and squared correlation coefficient (for regression).
                p_vals: a list of decision values or probability estimates (if '-b 1' 
                    is specified). If k is the number of classes, for decision values,
                    each element includes results of predicting k(k-1)/2 binary-class
                    SVMs. For probabilities, each element contains k values indicating
                    the probability that the testing instance is in each class.
                    Note that the order of classes here is the same as 'model.label'
                    field in the model structure.
        """
        X = np.asarray(X).reshape(1,-1)
        sys.stdout=StringIO() 
        p_lbl, p_acc, p_val = svm_predict([0], X.tolist(), self.svm)
        sys.stdout=bkp_stdout
        predicted_label = int(p_lbl[0])
        return [predicted_label, { 'p_lbl' : p_lbl, 'p_acc' : p_acc, 'p_val' : p_val }]
    
    def __repr__(self):        
        return "Support Vector Machine (kernel_type=%s, C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % (KERNEL_TYPE[self.param.kernel_type], self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree)



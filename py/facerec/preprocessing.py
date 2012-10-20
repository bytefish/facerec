import numpy as np
from facerec.feature import AbstractFeature
from facerec.util import asColumnMatrix
from scipy import ndimage
    
class HistogramEqualization(AbstractFeature):
    def __init__(self, num_bins=256):
        AbstractFeature.__init__(self)
        self._num_bins = num_bins
        
    def compute(self,X,y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp
        
    def extract(self,X):
        h, b = np.histogram(X.flatten(), self._num_bins, normed=True)
        cdf = h.cumsum()
        cdf = 255 * cdf / cdf[-1]
        return np.interp(X.flatten(), b[:-1], cdf).reshape(X.shape)
    
    def __repr__(self):
        return "HistogramEqualization (num_bins=%s)" % (self._num_bins)
        
class TanTriggsPreprocessing(AbstractFeature):
    def __init__(self, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
        AbstractFeature.__init__(self)
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)
    
    def compute(self,X,y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self,X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X,self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X,self._sigma1) - ndimage.gaussian_filter(X,self._sigma0))
        X = X / np.power(np.mean(np.power(np.abs(X),self._alpha)), 1.0/self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X),self._tau),self._alpha)), 1.0/self._alpha)
        X = self._tau*np.tanh(X/self._tau)
        return X

    def __repr__(self):
        return "TanTriggsPreprocessing (alpha=%.3f,tau=%.3f,gamma=%.3f,sigma0=%.3f,sigma1=%.3f)" % (self._alpha,self._tau,self._gamma,self._sigma0,self._sigma1)

from facerec.lbp import ExtendedLBP

class LBPPreprocessing(AbstractFeature):

    def __init__(self, lbp_operator = ExtendedLBP(radius=1, neighbors=8)):
        AbstractFeature.__init__(self)
        self._lbp_operator = lbp_operator
    
    def compute(self,X,y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self,X):
        return self._lbp_operator(X)

    def __repr__(self):
        return "LBPPreprocessing (lbp_operator=%s)" % (repr(self._lbp_operator))

from facerec.normalization import zscore, minmax

class MinMaxNormalizePreprocessing(AbstractFeature):
    def __init__(self, low=0, high=1):
        AbstractFeature.__init__(self)
        self._low = low
        self._high = high
        
    def compute(self,X,y):
        Xp = []
        XC = asColumnMatrix(X)
        self._min = np.min(XC)
        self._max = np.max(XC)
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp
    
    def extract(self,X):
        return minmax(X, self._low, self._high, self._min, self._max)
        
    def __repr__(self):
        return "MinMaxNormalizePreprocessing (low=%s, high=%s)" % (self._low, self._high)
        
class ZScoreNormalizePreprocessing(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        self._mean = 0.0 
        self._std = 1.0
        
    def compute(self,X,y):
        XC = asColumnMatrix(X)
        self._mean = XC.mean()
        self._std = XC.std()
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp
    
    def extract(self,X):
        return zscore(X,self._mean, self._std)

    def __repr__(self):
        return "ZScoreNormalizePreprocessing (mean=%s, std=%s)" % (self._mean, self._std)

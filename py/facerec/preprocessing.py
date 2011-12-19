import numpy as np
from facerec.feature import AbstractFeature
from facerec.util import asColumnMatrix
from scipy import ndimage
	
class HistogramEqualization(AbstractFeature):
	def __init__(self, num_bins=256):
		AbstractFeature.__init__(self)
		self._num_bins = num_bins
		
	def compute(self,X,y):
		return self.extract(X)
		
	def extract(self,X):
		h, b = histogram(X.flatten(), self._num_bins, normed=True)
		cdf = h.cumsum()
		cdf = 255 * cdf / cdf[-1]
		return interp(X.flatten(), b[:-1], cdf).reshape(X.shape)
		
class TanTriggsPreprocessing(AbstractFeature):
	def __init__(self, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
		AbstractFeature.__init__(self)
		self._alpha = alpha
		self._tau = tau
		self._gamma = gamma
		self._sigma0 = sigma0
		self._sigma1 = sigma1
	
	def compute(self,X,y):
		Xp = []
		for xi in X:
			Xp.append(self.extract(xi))
		return Xp

	def extract(self,X):
		X = np.power(X,self._gamma)
		X = np.asarray(ndimage.gaussian_filter(X,self._sigma1) - ndimage.gaussian_filter(X,self._sigma0))
		X = X / np.power(np.mean(np.power(np.abs(X),self._alpha)), 1.0/self._alpha)
		X = X / np.power(np.mean(np.power(np.minimum(np.abs(X),self._tau),self._alpha)), 1.0/self._alpha)
		X = 0.5*np.tanh(X/self._tau)+0.5
		return X

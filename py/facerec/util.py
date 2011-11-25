import numpy as np
import random
from scipy import ndimage

def normalize(X, low, high, minX=None, maxX=None, dtype=np.float32):
	""" min-max normalize a given matrix to given range [low,high].
	
	Args:
		X [rows x columns] input data
		low [numeric] lower bound
		high [numeric] upper bound
	"""
	X = np.array(X, dtype=dtype)
	if minX is None:
		minX = np.min(X)
	if maxX is None:
		maxX = np.max(X)
	# Normalize to [0...1].	
	X = X - minX
	X = X / (maxX - minX)
	# Scale to [low...high].
	X = X * (high-low)
	X = X + low
	return X

def zscore(X):
	"""
	Standard Normalization X = (X-mean(X))/std(X)
	"""
	X = np.asanyarray(X)
	mean = X.mean()
	std = X.std() 
	X = (X-mean)/std
	return X, mean, std
		
def shuffle(X,y):
	idx = np.argsort([random.random() for i in xrange(y.shape[0])])
	return X[:,idx], y[idx]
	
def tan_triggs(X, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
	""" Preprocessing steps as described in:
		
		"Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions"
		
	Args:
		alpha [numeric] 
		tau [numeric] 
		gamma [numeric] preferrably in range [0,0.5]
		sigma0 [numeric] inner kernel
		sigma1 [numeric] outer kernel
	"""
	# gamma correction
	X = np.power(X,gamma)
	# edges with a difference of gaussians
	X = np.asarray(ndimage.gaussian_filter(X,sigma1) - ndimage.gaussian_filter(X,sigma0))
	# approximated contrast equalization
	X = X / np.power(np.mean(np.power(np.abs(X),alpha)), 1.0/alpha)
	X = X / np.power(np.mean(np.power(np.minimum(np.abs(X),tau),alpha)), 1.0/alpha)
	# cut off extreme values (eyes, nostrils)
	X = tau*np.tanh(X/tau)
	return X

def range_f(begin, end, step):
	seq = []
	while True:
		if step == 0: break
		if step > 0 and begin > end: break
		if step < 0 and begin < end: break
		seq.append(begin)
		begin = begin + step
	return seq

def grid(grid_parameters):
	from itertools import product
	grid = []
	for parameter in grid_parameters:
		begin, end, step = parameter
		grid.append(range_f(begin, end, step))
	return product(*grid)


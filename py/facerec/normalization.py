import numpy as np

def minmax(X, low, high, dtype=np.float):
	X = np.asarray(X)
	minX, maxX = (np.min(X),np.max(X))
	# normalize to [0...1].	
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	return np.asarray(X,dtype=dtype)

def zscore(X, mean=None, std=None):
	X = np.asarray(X)
	if mean is None:
		mean = X.mean()
	if std is None:
		std = X.std()
	X = (X-mean)/std
	return X

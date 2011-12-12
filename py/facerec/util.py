import numpy as np
import random
from scipy import ndimage

def asRowMatrix(X):
	"""
	Creates a row-matrix from multi-dimensional data items in list l.
	
	X [list] List with multi-dimensional data.
	"""
	if len(X) == 0:
		return np.array([])
	total = 1
	for i in range(0, np.ndim(X[0])):
		total = total * X[0].shape[i]
	print total
	mat = np.empty([0, total], dtype=X[0].dtype)
	for row in X:
		print "add"
		mat = np.append(mat, row.reshape(1,-1), axis=0) # same as vstack
	return np.asmatrix(mat)

def asColumnMatrix(X):
	"""
	Creates a column-matrix from multi-dimensional data items in list l.
	
	X [list] List with multi-dimensional data.
	"""
	if len(X) == 0:
		return np.array([])
	total = 1
	for i in range(0, np.ndim(X[0])):
		total = total * X[0].shape[i]
	mat = np.empty([total, 0], dtype=X[0].dtype)
	for col in X:
		mat = np.append(mat, col.reshape(-1,1), axis=1)
	return np.asmatrix(mat)


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


def olbp(X):
	"""
	Create LBP image with fixed neighborhood
	"""
	X = np.asarray(X)
	X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
		+ (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
		+ (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
		+ (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
		+ (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
		+ (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
		+ (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
		+ (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
	return X

def clbp(X, radius=1, neighbors=8):
	X = np.asanyarray(X)
	ysize, xsize = X.shape
	# define circle
	angles = 2*np.pi/neighbors
	theta = np.arange(0,2*np.pi,angles)
	
	# calculate sample points on circle with radius
	sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
	sample_points *= radius
	
	# find boundaries of the sample points
	miny=min(sample_points[:,0])
	maxy=max(sample_points[:,0])
	minx=min(sample_points[:,1])
	maxx=max(sample_points[:,1])
	
	# calculate block size, each LBP code is computed within a block of size bsizey*bsizex
	blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
	blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
	
	# coordinates of origin (0,0) in the block
	origy =  0 - np.floor(min(miny,0))
	origx =  0 - np.floor(min(minx,0))
		
	# calculate output image size
	dx = xsize - blocksizex + 1
	dy = ysize - blocksizey + 1
	
	# get center points
	C = np.asarray(X[origy:origy+dy,origx:origx+dx], dtype=np.uint8)
	result = np.zeros((dy,dx), dtype=np.uint32)
	for i,p in enumerate(sample_points):
		# get coordinate in the block
		y,x = p + (origy, origx)
		
		# Calculate floors, ceils and rounds for the x and y.
		fx = np.floor(x)
		fy = np.floor(y)
		
		cx = np.ceil(x)
		cy = np.ceil(y)
		
		rx = round(x)
		ry = round(y)
		
		# calculate fractional part	
		ty = y - fy
		tx = x - fx
	
		# calculate interpolation weights
		w1 = (1 - tx) * (1 - ty)
		w2 =      tx  * (1 - ty)
		w3 = (1 - tx) *      ty
		w4 =      tx  *      ty
	
		# calculate interpolated image
		N = w1*X[fy:fy+dy,fx:fx+dx]
		N += w2*X[fy:fy+dy,cx:cx+dx]
		N += w3*X[cy:cy+dy,fx:fx+dx]
		N += w4*X[cy:cy+dy,cx:cx+dx]
		
		# compare to center pixels (returns boolean map)
		D = N >= C
	
		# add to result
		result += (1<<i)*D
	return result

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
	# cut off extreme values (eyes, nostrils) and scale between 0-1
	X = 0.5*np.tanh(X/tau)+0.5
	return X


# Implements various distance metrics (because my old scipy.spatial.distance module is horrible)
import numpy as np

######################################
# Euclidean Space Similarity
######################################
def euclidean(p, q):
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	return np.sqrt(np.sum(np.power((p-q),2)))

######################################
# Angle Dissimilarity
######################################
def cosine(p,q):
	"""
	Negated Mahalanobis Cosine Distance.
	
	Literature:
		"Studies on sensitivity of face recognition performance to eye location accuracy.". Master Thesis (2004), Wang
	"""
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	return -np.dot(p.T,q) / (np.sqrt(np.dot(p,p.T)*np.dot(q,q.T)))

######################################
# Histogram Distance Calculations
######################################

def chi_square(p,q):
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	bin_dists = (p-q)**2 / (p+q+np.finfo('float').eps)
	return np.sum(bin_dists)
	
def histogram_intersection(p,q):
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	return np.sum(np.minimum(p,q))

def brd(p,q):
	"""
	Calculates the Bin Ratio Dissimilarity.

	Args:
		p [1D vector] normalized histogram 
		h [1D vector] normalized histogram

	Literature:
	  "Use Bin-Ratio Information for Category and Scene Classification" (2010), Xie et.al. 
	"""
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	a = np.abs(1-np.dot(p,q.T)) # NumPy needs np.dot instead of * for reducing to tensor
	b = ((p-q)**2 + 2*a*(p*q))/((p+q)**2+np.finfo('float').eps)
	return np.abs(np.sum(b))
	
def l1_brd(p,q):
	"""
	Calculates the L1-Bin Ratio Dissimilarity.

	Args:
		p [1D vector] normalized histogram 
		h [1D vector] normalized histogram

	Literature:
	  "Use Bin-Ratio Information for Category and Scene Classification" (2010), Xie et.al. 
	"""
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	a = np.abs(1-np.dot(p,q.T)) # NumPy needs np.dot instead of * for reducing to tensor
	b = ((p-q)**2 + 2*a*(p*q)) * abs(p-q) / ((p+q)**2+np.finfo('float').eps)
	return np.abs(np.sum(b))
	
def chi_square_brd(p,q):
	"""
	Calculates the Chi-Square-Bin Ratio Dissimilarity.

	Args:
		p [1D vector] normalized histogram 
		h [1D vector] normalized histogram

	Literature:
	  "Use Bin-Ratio Information for Category and Scene Classification" (2010), Xie et.al. 
	"""
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	a = np.abs(1-np.dot(p,q.T)) # NumPy needs np.dot instead of * for reducing to tensor
	b = ((p-q)**2 + 2*a*(p*q)) * (p-q)**2 / ((p+q)**3+np.finfo('float').eps)
	return np.abs(np.sum(b))

# python imports
import operator as op

# numpy imports
from scipy.linalg import eig
import numpy as np

# facerec specific stuff
import facerec.distance as distance
from facerec.util import *

class Model(object):
	""" Model that must be subclassed for Cross Validation use. Pretty useless right now, probably more useful in later versions. """
	def __init__(self, name="generic"):
		self.name=name
	
	def save(self, filename="model.txt"):
		pass
	
	def load(self, filename="model.txt"):
		pass
	
	def __repr__(self):
		return "Model"


class PredictableModel(Model):
	"""
	Predictable model is a model which is able
	to predict given multi-dimensional data.
	
	"""
	def __init__(self, name):
		Model.__init__(self, name=name)
	
	def compute(self, X, y):
		"""
		Abstract method for computing a PredictableModel.
		
		Args:
		  X [list] list of multi-dimensional data items.
		  y [list] integer label for each data item
		"""
		pass
		
	def predict(self, X):
		"""
		Abstract method for predicting on a PredictableModel.
		
		Args:
		  X [array] multidimensional data
		Returns:
		  [int] predicted class
		"""
		pass
	
class PCA(Model):
	""" Performs a Principal Component Aanalysis.
	
	As described on http://en.wikipedia.org/wiki/Principal_component_analysis (last accessed 08/2011).
	
	Attributes:
		num_components [int] Number of components to include (default (columns(X) - 1)).
		L [1 x num_components] eigenvalues
		W [dim x num_components] eigenvectors
		mean [dim x 1] sample mean
	"""
	def __init__(self, X=None, num_components=0, ignore_components=0):
		Model.__init__(self, name="PCA")
		self.num_components = num_components
		self.ignore_components = ignore_components
		try:
			self.compute(X)
		except:
			pass
		
	
	def compute(self, X):
		if self.num_components <= 0 or (self.num_components > X.shape[1]-1):
			self.num_components = X.shape[1]-1 
			
		self._mean = X.mean(axis=1).reshape(-1,1)
		X = X - self._mean

		# allocates too much memory while computation
		self._eigenvectors, self._eigenvalues, variances = np.linalg.svd(X, full_matrices=False)
		
		# sort descending
		idx = np.argsort(-self._eigenvalues)
		self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
		
		# use only num_components
		self._eigenvectors = self._eigenvectors[0:,self.ignore_components:self.num_components].copy()
		self._eigenvalues = self._eigenvalues[self.ignore_components:self.num_components].copy()
		
		# calculate eigenvalues from singular values
		self._eigenvalues = np.power(self._eigenvalues,2) / X.shape[1]
	
	def __repr__(self):
		return "PCA (num_components=%d)" % (self.num_components)
		
	def project(self, X):
		X = X - self._mean
		return np.dot(self._eigenvectors.T, X)

	def reconstruct(self, X):
		X = np.dot(self._eigenvectors, X)
		return X + self._mean

	@property
	def W(self):
		return self._eigenvectors

	@property
	def mean(self):
		return self._mean
	
	@property
	def L(self):
		return self._eigenvalues
		
class LDA(Model):
	""" Performs a Multiclass Linear Discriminant Analysis.
	
	Implementation follows the description on http://en.wikipedia.org/wiki/Linear_discriminant_analysis (last accessed 08/2011).
	
	Attributes:
		num_components [int] Components in this projection.
		L [1 x num_components] Eigenvalues found by the LDA.
		W [num_components x num_data] Eigenvectors found by the LDA.
	"""
	
	def __init__(self, X=None, y=None, num_components=0):
		""" Initializes the LDA class and computes the LDA if data is given.
	
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
			num_components [int] components to include (default: len(unique(y))-1)
		
		"""
		Model.__init__(self, name="LDA")
		self.num_components = num_components
		if (X is not None) and (y is not None):
			self.compute(X,y)

	def compute(self, X, y):
		d = X.shape[0]
		c = len(np.unique(y))		
		if self.num_components <= 0:
			self.num_components = c-1
		elif self.num_components > (c-1):
			self.num_components = c-1
		
		meanTotal = X.mean(axis=1).reshape(-1,1)
		
		Sw = np.zeros((d, d), dtype=np.float32)
		Sb = np.zeros((d, d), dtype=np.float32)
		
		for i in range(0,c):
			Xi = X[:,np.where(y==i)[0]]
			meanClass = np.mean(Xi, axis = 1).reshape(-1,1)
			Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
			Sb = Sb + Xi.shape[1] * np.dot((meanClass - meanTotal), (meanClass - meanTotal).T)
		
		self._eigenvalues, self._eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)

		# sort descending by eigenvalue
		idx = np.argsort(-self._eigenvalues.real)
		self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
		
		# copy only the (c-1) non-zero eigenvalues
		self._eigenvalues = np.array(self._eigenvalues[0:self.num_components].real, dtype=np.float32, copy=True)
		self._eigenvectors = np.matrix(self._eigenvectors[0:,0:self.num_components].real, dtype=np.float32, copy=True)
		
	def project(self, X):
		return np.dot(self._eigenvectors.T, X)

	def reconstruct(self, X):
		return np.dot(self._eigenvectors, X)
		
	@property
	def W(self):
		return self._eigenvectors
	
	@property
	def L(self):
		return self._eigenvalues
	
	def __repr__(self):
		return "LDA (num_components=%d)" % (self.num_components)
		
class NearestNeighbor(PredictableModel):

	def __init__(self, X=None, y=None, dist_metric=distance.euclidean, k=1):
		PredictableModel.__init__(self, name="NearestNeighbor")
		self.k = k
		self.dist_metric = dist_metric
		if (X is not None) and (y is not None):
			self.compute(X,y)
	
	def compute(self, X, y):
		self.X = X
		self.y = y
		
	def predict(self, q):
		distances = []
		for xi in self.X:
			xi = xi.reshape(-1,1)
			d = self.dist_metric(xi, q)
			distances.append(d)
		if len(distances) > len(self.y):
			raise Exception("More distances than classes. Is your distance metric correct?")
		idx = np.argsort(np.array(distances))
		sorted_y = self.y[idx]
		sorted_y = sorted_y[0:self.k]
		hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
		return max(hist.iteritems(), key=op.itemgetter(1))[0]

class Eigenfaces(PredictableModel):
	""" Implements the Eigenfaces method by Pentland and Turk.
	
	For detailed algorithmic analysis refer to paper:

		@article{TP1991,
			author = {Turk, M. and Pentland, A.},
			journal = {Journal of Cognitive Neuroscience},
			number = {1},
			pages = {71--86},
			title = {{Eigenfaces for Recognition}},
			volume = {3},
			year = {1991}
		}	
				
	Attributes:
		P: Projected Training dataset (data given in columns).
		y: Classes corresponding to samples in P.
		W: Eigenvectors of this projection.
		k: Number of Nearest Neighbor used in prediction.
	"""
	
	def __init__(self, classifier=NearestNeighbor(), X=None, y=None, num_components=0, ignore_components=0):
		""" Initialize Eigenfaces object and computes Eigenfaces if data was given.
		
		Args:
			X [dim x num_data] Multidimensional list with data given in columns.
			y [1 x num_data] List with classes corresponding to the samples of X.
			num_components [int] Number of components to use in PCA projection.
			ignore_components [int] Number of components to ignore in PCA projection.
		"""
		PredictableModel.__init__(self, name="Eigenfaces")
		self.num_components = num_components
		self.ignore_components = ignore_components
		self.classifier = classifier
		if (X is not None) and (y is not None):
			self.compute(X,y)
			
	def compute(self, X, y):
		""" Computes eigenfaces and projects X onto the num_components principal components.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
		"""
		M = asColumnMatrix(X)
		self.pca = PCA(M, num_components=self.num_components, ignore_components=self.ignore_components)
		self.num_components = self.pca.num_components
		
		# learn a classifier with the projections
		projections = []
		for x in X:
			xp = self.project(x.reshape(-1,1))
			projections.append(xp)
		self.y = y
		self.P = projections
		self.classifier.compute(projections, y)
		
	def project(self,X):
		return self.pca.project(X)
		
	def reconstruct(self, X):
		return self.pca.reconstruct(X)
			
	def predict(self, x):
		x = np.asarray(x).reshape(-1,1)
		q = self.project(x)
		return self.classifier.predict(q)

	def __repr__(self):
		return "Eigenfaces (classifier=%s, num_components=%d)" % (self.classifier.name, self.num_components)

	@property
	def W(self):
		return self.pca.W
		
	@property
	def L(self):
		return self.pca.L
		
	def empty(self):
		self.pca.empty()

class Fisherfaces(PredictableModel):
	""" Implements the Fisherface method.
	
	For detailed algorithmic analysis refer to paper:
	
		@article{BHK1997,
			author = {Belhumeur, P. N. and Hespanha, J. P. and Kriegman, D. J.},
			title = {{Eigenfaces vs. Fisherfaces: recognition using class specific linear projection}},
			journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
			month = jul,
			number = {7},
			pages = {711--720},
			publisher = {IEEE Computer Society},
			volume = {19},
			year = {1997}
		}
		
	Attributes:
		P [num_components x num_data] projections
		y [1 x num_data] classes
		W [dim x num_components] Eigenvectors of this projection.
		k [int] Number of Nearest Neighbor used in prediction.
	"""

	def __init__(self, X=None, y=None, classifier=NearestNeighbor(), num_components=0):
		""" Initializes Fisherfaces a model.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes (optional)
			classifier [PredictableModel] classifier 
		"""
		PredictableModel.__init__(self, name="Fisherfaces")
		self.num_components = num_components
		self.classifier = classifier
		if (X is not None) and (y is not None):
			self.compute(X,y)	
	
	def compute(self, X, y):
		""" Compute the Fisherfaces as described in [BHK1997]: Wpcafld = Wpca * Wfld.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
		"""
		M = asColumnMatrix(X) # reshape to NumPy matrix
		y = np.asarray(y) # Convert to NumPy array
		n = len(y)
		c = len(np.unique(y))
		# remove null space by projecting into n-c dimensions
		pca = PCA(M, n-c)
		# calculate LDA
		lda = LDA(pca.project(M), y, self.num_components)
		# Fisherfaces = Wpca * Wlda
		self._eigenvectors = np.dot(pca.W,lda.W)
		# store projection and classes
		self.num_components = lda.num_components
		# learn the classifier on the data
		projections = []
		for x in X:
			p = self.project(x.reshape(-1,1))
			projections.append(p)
		self.classifier.compute(projections, y)
		
	def project(self, X):
		return np.dot(self._eigenvectors.T, X)
	
	def reconstruct(self, X):
		return np.dot(self._eigenvectors, X)

	def predict(self, X):
		X = np.asarray(X).reshape(-1,1)
		q = self.project(X)
		return self.classifier.predict(q)
	
	@property
	def W(self):
		return self._eigenvectors

	def __repr__(self):
		return "Fisherfaces (classifier=%s, num_components=%s)" % (self.classifier.name, self.num_components)


class LBP(PredictableModel):
	""" Implements Local Binary Pattern as described in 
	
	@CONFERENCE{Ahonen2004,
		author = {Ahonen, Timo and Hadid, Abdenour and Pietikainen, Matti},
		title = {Face Recognition with Local Binary Patterns},
		booktitle = {Proceedings of European Conference on Computer Vision},
		year = {2004},
	}
	
	This is a Python version of the MATLAB implementation by Marko Heikkilae 
	and Timo Ahonen, available at \url{http://www.cse.oulu.fi/MVG/Downloads/LBPMatlab}. 
	
	All credit goes to them.
	"""
	def __init__(self, lbp_operator=clbp, classifier=NearestNeighbor(), radius=1, neighbors=8, sz = (8,8)):
		"""
		Args:
		  lbp_operator []
		  classifier [Model]
		  radius [int]
		  neighbors [int]
		  sz [tuple] 
		"""
		PredictableModel.__init__(self, name="Local Binary Patterns")
		self.lbp_operator = lbp_operator
		self.classifier = classifier
		self.radius = radius
		self.neighbors = neighbors
		self.sz = sz
		
	def compute(self,X,y):
		"""
		Compute LBP patterns for given data.
		
		Args:
			X [list] multi-dimensional input data
			y [1 x num_data] Classes
		
		"""
		P = []
		for i in range(0,len(X)):
			H = self.spatially_enhanced_histogram(X[i])
			P.append(H)
		self.classifier.compute(P,y)
	
	def spatially_enhanced_histogram(self, X):
		L = self.lbp_operator(X, radius = self.radius, neighbors = self.neighbors)
		# build the histogram vector
		lbp_height, lbp_width = L.shape
		grid_rows, grid_cols = self.sz
		py = int(np.floor(lbp_height/grid_rows))
		px = int(np.floor(lbp_width/grid_cols))
		E = []
		for row in range(0,grid_rows):
			for col in range(0,grid_cols):
				C = L[row*py:(row+1)*py,col*px:(col+1)*px]
				H = np.histogram(C, bins=2**self.neighbors, range=(0, 2**self.neighbors), normed=True)[0]
				# probably useful to apply a mapping?
				E.extend(H)
		return np.asarray(E)
	
	def predict(self, X):
		q = self.spatially_enhanced_histogram(X)
		return self.classifier.predict(q)
	
	def __repr__(self):
		return "Local Binary Pattern (classifier=%s, radius=%s, neighbors=%s, grid=%s)" % (self.classifier.name, self.radius, self.neighbors, str(self.sz))

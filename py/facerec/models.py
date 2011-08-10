import operator as op
import numpy as np
from scipy.linalg import eig
"""
	Author: philipp <bytefish[at]gmx.de>
	License: BSD (see LICENSE for details)
	Description:
		Models used for Data Analysis, Validation or Visualization.
"""

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
		
		
class PCA(Model):
	""" Performs a Principal Component Aanalysis.
	
	As described on http://en.wikipedia.org/wiki/Principal_component_analysis (last accessed 08/2011).
	
	Attributes:
		num_components [int] Number of components to include (default (columns(X) - 1)).
		L [1 x num_components] eigenvalues
		W [dim x num_components] eigenvectors
		mean [dim x 1] sample mean
	"""
	def __init__(self, X=None, num_components=None, ignore_components=None):
		Model.__init__(self, name="PCA")
		self.num_components = num_components
		self.ignore_components = ignore_components
		try:
			self.compute(X)
		except:
			pass
		
	
	def compute(self, X):
		""" Computes the Eigenvalues of (mean-subtracted) X and selects num_components largest eigenvectors corresponding to their eigenvalue (== PCA).

		Args:
			X [dim x num_data] input data
		"""
		if self.num_components is None:
			self.num_components = X.shape[1]-1 
		if self.ignore_components is None:
			self.ignore_component = 0
			
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
		
		self._eigenvalues = np.power(self._eigenvalues,2) / X.shape[1]
	
	def __repr__(self):
		return "PCA"
		
	def project(self, X):
		""" Projects a sample into the PCA subspace.
		
		Args:
			X [dim x cols]: sample(s) to project
			
		Returns:
			Xhat = X - mean(X,2);
			W'*Xhat;
		"""
		X = X - self._mean
		return np.dot(self._eigenvectors.T, X)

	def reconstruct(self, X):
		""" Reconstructs a given column vector.
		
		Args: 
			X [num_components x cols] projection(s) to reconstruct
			
		Returns:
			Xhat = W*X
			X = Xhat + mean 
		"""
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
	
	def empty(self):
		""" Empty the current model. """
		self._eigenvalues = []
		self._eigenvectors = []
		self._mean = []

		
class LDA(Model):
	""" Performs a Multiclass Linear Discriminant Analysis.
	
	Implementation follows the description on http://en.wikipedia.org/wiki/Linear_discriminant_analysis (last accessed 08/2011).
	
	Attributes:
		num_components [int] Components in this projection.
		L [1 x num_components] Eigenvalues found by the LDA.
		W [num_components x num_data] Eigenvectors found by the LDA.
	"""
	
	def __init__(self, X=None, y=None, num_components=None):
		""" Initializes the LDA class and computes the LDA if data is given.
	
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
			num_components [int] components to include (default: len(unique(y))-1)
		
		"""
		Model.__init__(self, name="LDA")
		self.num_components = num_components
		try:
			self.compute(X,y)
		except:
			pass

	def compute(self, X, y):
		""" Performs a LDA.
	
		Args:
			X [dim x num_data] Multidimensional list with data given in columns.
			y [1 x num_data] List with classes corresponding to the samples of X.
		"""
		d = X.shape[0]
		c = len(np.unique(y))
		
		if self.num_components is None:
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
		""" Projects X onto the num_components found by the LDA: W'*X
	
		Args:
			X [dim x cols] sample(s) to project
		Returns:
			LDA Projection
		"""
		return np.dot(self._eigenvectors.T, X)

	def reconstruct(self, X):
		""" LDA Reconstruction 
		
		Args:
			X [num_components x cols] projection(s) to reconstruct
		Returns:
			W*X
		"""
		return np.dot(self._eigenvectors, X)
	
	def __repr__(self):
		return "LDA"
	
	@property
	def W(self):
		return self._eigenvectors
	
	@property
	def L(self):
		return self._eigenvalues
		
	def empty(self):
		pass
		#""" Empty the current model.	"""
		self._eigenvectors = []
		self._eigenvalues = []
		
class NearestNeighbor(object):
	""" Implements k-Nearest Neighbors. 
	
	Euclidean distance is used for measuring neighborhood. Please add your distance metric if necessary.
	"""
	@staticmethod
	def predict(P, Q, y, k=1):
		""" k-Nearest Neighbor
		
		Args:
			P [dim x num_data] Reference vectors. 
			Q [dim x 1] Query vector.
			y [1 x num_data] Classes corresponding to the samples of P.
			k [int] Number of neighbors for this prediction (default 1).
		Returns:
			Predicted class given by the majority of k neighbors.
		"""
		Q = Q.reshape(-1,1)
		distances = np.sqrt(np.power((P-Q),2).sum(axis=0)).flat
		idx = np.argsort(np.array(distances))
		y = y[idx]
		y = y[0:(k+1)]
		hist = dict((key,val) for key, val in enumerate(np.bincount(y)) if val)
		return max(hist.iteritems(), key=op.itemgetter(1))[0]
			
class Eigenfaces(Model):
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
	
	def __init__(self, X=None, y=None, num_components=None, ignore_components=None,k=1):
		""" Initialize Eigenfaces object and computes Eigenfaces if data was given.
		
		Args:
			X [dim x num_data] Multidimensional list with data given in columns.
			y [1 x num_data] List with classes corresponding to the samples of X.
			num_components [int] Number of components to use in PCA projection.
			ignore_components [int] Number of components to ignore in PCA projection.
		"""
		Model.__init__(self, name="Eigenfaces")
		self.num_components = num_components
		self.ignore_components = ignore_components
		self.k=k
		try:
			self.compute(X,y)
		except:
			pass
			
	def compute(self, X, y):
		""" Computes eigenfaces and projects X onto the num_components principal components.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
		"""
		self.pca = PCA(X, num_components=self.num_components, ignore_components=self.ignore_components)
		self.y = y
		self.P = self.project(X)
		
	def project(self,X):
		""" Projects X onto the num_components Principal Components.
		
		Args:
			X [dim x cols] sample(s) to project
		Returns:
			PCA projection
		"""
		return self.pca.project(X)
		
	def reconstruct(self, X):
		""" Reconstruct X from the PCA projection.
		
		Args:
			X [num_components x cols] projection(s) to reconstruct
			
		Returns:
			PCA reconstruction		
		"""
		return self.pca.reconstruct(X)
		
	def __repr__(self):
		return "Eigenfaces"
	
	def predict(self, X):
		""" k-Nearest Neighbor prediction of given column vector.
		
		Args:
			X [dim x 1] sample to predict
			
		Returns:
			Predicted class given by the majority of the k nearest neighbors.
		"""
		Q = self.project(X)
		return NearestNeighbor.predict(self.P, Q, self.y, k=self.k)
	
	@property
	def W(self):
		return self.pca.W
		
	@property
	def L(self):
		return self.pca.L
		
	def empty(self):
		self.pca.empty()
		self.P = []
		self.y = []
		

class Fisherfaces(Model):
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

	def __init__(self, X=None, y=None, num_components=None, k=1):
		""" Initializes Fisherfaces a model.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes (optional)
			k [int] Number of Nearest Neighbor used in prediction (optional)
		"""
		Model.__init__(self, name="Fisherfaces")
		self.k = k
		self.num_components = num_components
		try:
			self.compute(X,y)
		except:
			pass
	
	
	def compute(self, X, y):
		""" Compute the Fisherfaces as described in [BHK1997]: Wpcafld = Wpca * Wfld.
		
		Args:
			X [dim x num_data] input data
			y [1 x num_data] classes
		"""
		n = len(y)
		c = len(np.unique(y))
		# remove null space by projecting into n-c dimensions
		pca = PCA(X, n-c)
		# calculate LDA
		lda = LDA(pca.project(X), y, self.num_components)
		# Fisherfaces = Wpca * Wlda
		self._eigenvectors = pca.W*lda.W
		# store projection and classes
		self.num_components = lda.num_components
		self.P = self.project(X)
		self.y = y
		
		
		
	def project(self, X):
		""" Project X into Subspace.
		
		Args:
			X [dim x cols] sample(s) to project.
		
		Returns:
			Projection W'*X
		"""
		return np.dot(self._eigenvectors.T, X)
	
	def reconstruct(self, X):
		""" Reconstruct X from Subspace.
		
		Args:
			X [num_components x cols] projection(s) to reconstruct
			
		Returns:
			Reconstruction W*X
		"""
		return np.dot(self._eigenvectors, X)

	def predict(self, X):
		""" Find the k-Nearest Neighbor of this instance.
		
		Args:
			X [dim x 1] sample to predict
		
		Returns:
			Predicted class given by the majority of the k nearest neighbors. 
		"""
		Q = self.project(X)
		return NearestNeighbor.predict(self.P, Q, self.y, k=self.k)
	
	@property
	def W(self):
		return self._eigenvectors
		
	def empty(self):
		""" Empty the current model (free some space). """
		self._eigenvectors = []
		self.P = []
		
	def __repr__(self):
		return "Fisherfaces (num_components=%s)" % (self.num_components)
		
if __name__ == "__main__":
	# Nearest Neighbor
	P = np.array([[0,0],[0.1,0.3],[0,1],[1,1],[9,9],[9.5,9.5],[9,8],[10,10]]).T
	Q = np.array([[9],[9]])
	y = np.array([0, 0, 0, 0, 1, 1 , 1 ,1])
	prediction = NearestNeighbor.predict(P, Q, y,k=1)
	print prediction
	# see http://www.bytefish.de/wiki/pca_lda_with_gnu_octave for this example
	X = np.array([[2,3],[3,4],[4,5],[5,6],[5,7],[2,1],[3, 2],[4 ,2],[4 ,3],[6, 4],[7,6]]).T
	y = np.array([ 0,0,0,0,0,1,1,1,1,1,1])
	
	# PCA
	pca = PCA(X)
	print pca.W
	# LDA
	lda = LDA(X,y)
	print lda.W

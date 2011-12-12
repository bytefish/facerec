import numpy as np
import logging
import math

# libsvm: svm implementation used
from svmutil import *

# facerec specific stuff
from facerec.models import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.util import asRowVector

# for suppressing output
import sys
from StringIO import StringIO

# function handle to stdout
bkp_stdout=sys.stdout

def range_f(begin, end, step):
	"""
	Equivalent to range, but also accepts floats (from libsvm/tools/grid.py).
	"""
	seq = []
	while True:
		if step == 0: break
		if step > 0 and begin > end: break
		if step < 0 and begin < end: break
		seq.append(begin)
		begin = begin + step
	return seq

def grid(grid_parameters):
	"""
	Builds the parameter space to search the best parameter combination ( == Cartesian product of parameter ranges).
	
	Args:
	  grid_parameters [list] list of tuples defining the parameter space
	Example:
	  grid_iter = grid([(PARAM1_BEGIN,PARAM1_END,PARAM1_STEP), (PARAM2_BEGIN,PARAM2_END,PARAM2_STEP)])
	"""
	from itertools import product
	grid = []
	for parameter in grid_parameters:
		begin, end, step = parameter
		grid.append(range_f(begin, end, step))
	return product(*grid)

class SVM(PredictableModel):
	"""
	This class is just a simple wrapper to use libsvm in the 
	CrossValidation module. If you don't use this framework
	use the validation methods coming with LibSVM, they are
	much easier to access (simply pass the correct class 
	labels in svm_predict and you are done...).
	
	The grid search method in this class is somewhat similar
	to libsvm grid.py, as it performs a parameter search over
	a logarithmic scale.	Again if you don't use this framework, 
	use the libsvm tools as they are much easier to access.
	
	Please keep in mind to normalize your input data, as expected
	for the model. There's no way to assume a generic normalization
	step.
	
	Atttributes:
		svm [svm.svm_model] model this svm is based on
	"""
	
	def __init__(self, X=None, y=None, param=svm_parameter("-q")):
		"""
		Args:
			param [svm_parameter] parameters to use for this SVM model
		"""
		PredictableModel.__init__(self, name="SVM")
		self.logger = logging.getLogger("facerec.models.SVM")
		self.param = param
		self.svm = svm_model()
		self.param = param
		if (X is not None) and (y is not None):
			self.compute(X,y)
		
	def compute(self, X, y):
		"""
		
		Args:
			X [list] list of feature vectors
			y [1 x num_data] corresponding class labels
		"""
		# log
		self.logger.info("SVM TRAIN (C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % (self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree))
		# turn data into a row vector (needed for libsvm)
		X = asRowVector(X)
		y = np.asanyarray(y)
		# svm_problem expects row vectors, so transpose
		problem = svm_problem(y, X.tolist())		
		# finally learn the svm
		self.svm = svm_train(problem, self.param)
		self.y = y
		
	def predict(self, X):
		"""
		Predicts a given Feature Vector in this SVM instance.
		"""
		# svm_predict expects row vector
		X = np.asarray(X).reshape(1,-1)
		## TODO Remove this hack to avoid libsvm output (by recompiling libsvm for example)
		sys.stdout=StringIO() 
		p_lbl, p_acc, p_val = svm_predict([0], X.tolist(), self.svm)
		sys.stdout=bkp_stdout
		return p_lbl[0]
	
	def grid_search(self, X, y, C_range=(-5,  15, 2), gamma_range=(3, -15, -2), k=5, num_cores=1):
		"""
		Performs a grid search in a logarithmic scale for the current SVM.
		
		Args:
		  X [list] list of multi-dimensional data items (will be reshaped into a feature vector!!).
		  y [list] list of corresponding classes
		  C_range [tuple] defines the range of the Cost parameter
		  gamma_range [tuple] defines the range of gamma
		  k [int] number of folds in cross validating a parameter combination
		  num_cores [int] number of available CPU cores (unused in this version)
		Returns:
			[svm_parameter] best parameter combination
		
		Example:
			svm.grid_search(X,y,C_range=(-1,3,1),gamma_range=(-1,3,1))
		 
		  # for learning with a RBF Kernel
		  from svmutil import *
		  
		  p = svm_parameter("-q")
		  p.kernel_type=RBF
		  svm.grid_search(X,y,C_range=(-1,3,1),gamma_range=(-1,3,1), param=p)
		  
		TODO
		  * Parallel Grid Search (see libsvm/tools/grid.py)
		  * Easier (non-libsvm dependent) parameter passing
		"""
		
		logger = logging.getLogger("facerec.models.SVM")
		logger.info("Performing a Grid Search.")

		# best parameter combination to return
		best_parameter = svm_parameter("-q")
		best_parameter.kernel_type = self.param.kernel_type
		best_parameter.nu = self.param.nu
		best_parameter.coef0 = self.param.coef0
		
		# either no gamma given or kernel is linear (only C to optimize)
		if (gamma_range is None) or (self.param.kernel_type == LINEAR):
			gamma_range = (self.param.gamma, self.param.gamma, self.param.gamma+1)
		
		# best validation error so far
		best_accuracy = np.finfo('float').max
		
		# create grid (cartesian product of ranges)		
		g = grid([C_range, gamma_range])
		acc = []
		for p in g:
			C, gamma = p
			C, gamma = 2**C, 2**gamma
			self.param.C, self.param.gamma = C, gamma

			# build model with new parameter
			s = SVM(param=self.param)

			# perform a k-fold cross validation
			cv = KFoldCrossValidation(model=s,k=k)
			cv.validate(X,y)

			# append parameter into list with accuracies for all parameter combinations
			acc.append([cv.accuracy, C, gamma])
			
			# store best parameter combination
			if cv.accuracy > best_accuracy:
				best_accuracy = cv.accuracy
				best_parameter.C, best_parameter.gamma = C, gamma
			
			self.logger.info("%d-CV Result = %.2f." % (k, cv.accuracy))

		# set best parameter combination to best found
		self.param = best_parameter
		return best_parameter, acc
		
	def __repr__(self):		
		return "Support Vector Machine (classifier=%s, kernel_type=%s, C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % ("SVM", KERNEL_TYPE[self.param.kernel_type], self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree)

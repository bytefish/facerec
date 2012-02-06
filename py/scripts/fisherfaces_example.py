from facerec.dataset import DataSet
from facerec.feature import *
from facerec.distance import *
from facerec.classifier import NearestNeighbor, SVM
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import plot_eigenvectors
from facerec.preprocessing import *
from facerec.operators import ChainOperator
from facerec.svm import grid_search
from facerec.lbp import *

import numpy as np
import logging,sys
# set up a handler for logging
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add handler to facerec modules
logger = logging.getLogger("facerec")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# load a dataset
dataSet = DataSet("/home/philipp/facerec/data/c2", sz=(100,100))
# define Fisherfaces as feature extraction method
feature = Identity()
#from svmutil import *
# define a SVM with RBF Kernel as classifier
#param = svm_parameter("-q")
#param.kernel_type = RBF
#classifier = SVM(param=param)
# define a 1-NN classifier with Euclidean Distance
classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=5)
# define the model as the combination
model = PredictableModel(feature=feature, classifier=classifier)

#best_parameter, results = grid_search(model, dataSet.data, dataSet.labels, k=5)
#print results
# show fisherfaces
#model.compute(dataSet.data, dataSet.labels)
#plot_eigenvectors(model.feature.eigenvectors, 1, sz=dataSet.data[0].shape, filename="/home/philipp/01.png")
#plot_eigenvectors(model.feature.model2.eigenvectors, 1, sz=(dataSet.data[0].shape[0]-2,dataSet.data[0].shape[1]-2), filename="/home/philipp/01.png")
# perform a 5-fold cross validation
cv = KFoldCrossValidation(model, k=10)
cv.validate(dataSet.data, dataSet.labels)
print cv

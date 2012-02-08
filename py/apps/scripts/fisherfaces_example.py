import sys
# append facerec to module search path
sys.path.append("../..")

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
# load a dataset (e.g. AT&T Facedatabase)
dataSet = DataSet("/home/philipp/facerec/data/at")
# define Fisherfaces as feature extraction method
feature = Fisherfaces()
# define a 1-NN classifier with Euclidean Distance
classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
# define the model as the combination
model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
model.compute(dataSet.data, dataSet.labels)
plot_eigenvectors(model.feature.eigenvectors, 10, sz=dataSet.data[0].shape, filename="fisherfaces.pdf")
# perform a 10-fold cross validation
cv = KFoldCrossValidation(model, k=10)
cv.validate(dataSet.data, dataSet.labels)
print cv

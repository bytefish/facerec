from facerec.dataset import DataSet
from facerec.feature import LBP,Fisherfaces
from facerec.distance import EuclideanDistance, CosineDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import plot_eigenvectors
from facerec.preprocessing import MinMaxNormalizePreprocessing
from facerec.operators import ChainOperator

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
dataSet = DataSet("/home/philipp/facerec/data/c1")
# define a 1-NN classifier with Euclidean Distance
classifier = NearestNeighbor(dist_metric=EuclideanDistance())
# define Fisherfaces as feature extraction method
feature = Fisherfaces()
# now stuff them into a PredictableModel
model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
model.compute(dataSet.data,dataSet.labels)
plot_eigenvectors(model.feature.eigenvectors, 9, sz=dataSet.data[0].shape)
# perform a 5-fold cross validation
cv = KFoldCrossValidation(model, k=5)
cv.validate(dataSet.data, dataSet.labels)

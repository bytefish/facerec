from facerec.models import *
from svmutil import *
from facerec.svm import *
from facerec.filereader import *
from facerec.validation import *
from facerec.util import *
import numpy as np
import logging,sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# get logging output from all modules
logger = logging.getLogger("facerec")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

d = DataSet("/home/philipp/facerec/data/yalefaces_recognition")

nn_cosine = NearestNeighbor(dist_metric=distance.euclidean)
fisherfaces = Fisherfaces(classifier=nn_cosine, num_components=40)
lbp = LBP(classifier=fisherfaces)
cv = KFoldCrossValidation(lbp,k=3)
cv.validate(d.data, d.labels, print_debug=True)

# prepare data for SVM
X = d.data
y = d.labels

# normalize the dataset
for i in range(0,len(X)):
	X[i] = normalize(X[i],0,1,0,255)

# define RBF kernel
p = svm_parameter("-q")
p.kernel_type=RBF
svm = SVM(param=p)
# optimize kernel parameters with a grid search
best_par, acc = svm.grid_search(X, y)


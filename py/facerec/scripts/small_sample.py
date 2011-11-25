import numpy as np
import matplotlib.pyplot as plt
 
from validation import *
from filereader import *
from models import *

dataset = FilesystemReader("/home/philipp/facerec/data/at")
dataset.shuffle() # always a good idea

m0 = Eigenfaces(num_components=50)
m1 = Fisherfaces()

# SimpleValidation expects you to generate the training/test folds
cv0 = SimpleValidation(m0)
cv1 = SimpleValidation(m1)

for i in range(2,10):
	trainIdx = []
	testIdx = []
	for j in range(0,len(np.unique(dataset.classes))):
		idx = np.where(dataset.classes==j)[0]
		trainIdx.extend(idx[0:i])
		testIdx.extend(idx[i:])
	cv0.validate(dataset.data,dataset.classes,trainIdx,testIdx)
	cv1.validate(dataset.data,dataset.classes,trainIdx,testIdx)
	
r0 = np.asarray(cv0.tp, dtype=np.float32)/np.asarray((cv0.tp+cv0.fp), dtype=np.float32)
r1 = np.asarray(cv1.tp, dtype=np.float32)/np.asarray((cv1.tp+cv1.fp), dtype=np.float32)

#---------------
# Plotting 
#---------------
filename="/home/philipp/facerec/at_database_vs_accuracy_xy.png"
t = np.arange(2., 10., 1.)
fig = plt.figure()
plt.plot(t, r0, 'k--', t, r1, 'k')
plt.legend(("Eigenfaces", "Fisherfaces"), 'lower right', shadow=True, fancybox=True)
plt.ylim(0,1)
plt.ylabel('Recognition Rate')
plt.xlabel('Database Size (Images per Person)')
fig.savefig(filename, format="png", transparent=False)
plt.show()

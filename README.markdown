# facerec #

## facerec/py/facerec ##

This project implements a face recognition framework for Python with:

* Preprocessing
    * Histogram Equalization
    * Local Binary Patterns
    * TanTriggsPreprocessing (Tan, X., and Triggs, B. "Enhanced local texture feature sets for face recognition under difficult lighting conditions.". IEEE Transactions on Image Processing 19 (2010), 1635–650.)
* Feature Extraction
    * Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
    * Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
    * Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)
        * Original LBP
        * Extended LBP
* Classifier
    * k-Nearest Neighbor; available distance metrics
        * Euclidean Distance
        * Cosine Distance
        * ChiSquare Distance
        * Bin Ratio Distance
    * Support Vector Machines; using libsvm bindings. (Vapnik, V. "Statistical Learning Theory.". John Wiley and Sons, New York, 1998.)
* Cross Validation
    * k-fold Cross Validation
    * Leave-One-Out Cross Validation
    * Leave-One-Class-Out Cross Validation

### A simple example for beginners

Due to popular request, I've created a simple example for getting started with the facerec framework.

#### Getting the data right 

We aren't doing a toy example, so you'll need some image data. For sake of simplicity I have assumed, that the images (the faces, persons you want to recognize) are given in folders. So imagine I have a folder `images` (the dataset!), with the subfolders person1, person2 and so on:

```
philipp@mango:~/facerec/data/images$ tree -L 2 | head -n 20
.
|-- person1
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- 3.jpg
|   |-- 4.jpg
|-- person2
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- 3.jpg
|   |-- 4.jpg

[...]
```

One of the public available datasets, that is already coming in such a folder structure is the AT&T Facedatabase, available at:

* [http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

Once unpacked it is going to look like this (on my filesystem it is unpacked to /home/philipp/facerec/data/at/, your path is different!):

```
philipp@mango:~/facerec/data/at$ tree .
.
|-- README
|-- s1
|   |-- 1.pgm
|   |-- 2.pgm
[...]
|   `-- 10.pgm
|-- s2
|   |-- 1.pgm
|   |-- 2.pgm
[...]
|   `-- 10.pgm
|-- s3
|   |-- 1.pgm
|   |-- 2.pgm
[...]
|   `-- 10.pgm

...

40 directories, 401 files
```

That's all that needs to be done.

#### Example: Fisherfaces

The following code listing now will learn a Fisherfaces model on the AT&T Facedatabase. I wrote a simple method `read_images`, which reads the images from a given path (and optionally resizes them). Make sure you have the folder structured as described above. The `read_images` method returns [X,y] being:

* **X**: A list of NumPy arrays (images).
* **y**: A list of integers (corresponding labels).

The source code listing is also in this github repository at:

* [facerec/py/apps/scripts/simple_example.py](https://github.com/bytefish/facerec/blob/master/py/apps/scripts/simple_example.py)

#### simple_example.py

```python
#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner <bytefish[at]gmx[dot]de>.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(self.sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Define the model as the combination
    model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    model.compute(X, y)
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
	    e = model.feature.eigenvectors[:,i].reshape(X[0].shape)
	    E.append(minmax_normalize(e,0,255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="fisherfaces.png")
    # Perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)
    # And print the result:
    print cv
```

#### Results

Since the AT&T Facedatabase is a fairly easy database we have got a `95.5%` recognition rate with the Fisherfaces method (with a 10-fold cross validation):

```
philipp@mango:~/github/facerec/py/apps/scripts$ python simple_example.py /home/philipp/facerec/data/at
2012-08-01 23:01:16,666 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 1/10.
2012-08-01 23:01:29,921 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 2/10.
2012-08-01 23:01:43,666 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 3/10.
2012-08-01 23:01:57,335 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 4/10.
2012-08-01 23:02:10,615 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 5/10.
2012-08-01 23:02:23,936 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 6/10.
2012-08-01 23:02:37,398 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 7/10.
2012-08-01 23:02:50,724 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 8/10.
2012-08-01 23:03:03,808 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 9/10.
2012-08-01 23:03:17,042 - facerec.validation.KFoldCrossValidation - INFO - Processing fold 10/10.

k-Fold Cross Validation (model=PredictableModel (feature=Fisherfaces (num_components=39), classifier=NearestNeighbor (k=1, dist_metric=EuclideanDistance)), k=10, runs=1, accuracy=95.50%, std(accuracy)=0.00%, tp=382, fp=18, tn=0, fn=0)
```

And we can have a look at the Fisherfaces found by the model:

![fisherfaces_at](https://github.com/bytefish/facerec/raw/master/py/apps/scripts/fisherfaces_at.png "fisherfaces_at")

### Advanced Examples: Building your own PredictableModel

Basically all face recognition algorithms are the combination of a [feature extraction](https://github.com/bytefish/facerec/blob/master/py/facerec/feature.py)
and a [classifier](https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py). The Eigenfaces method for example is a Principal Component Analysis with a Nearest Neighbor classifier. Local Binary Patterns Histograms . The feature (which must be an [AbstractFeature](https://github.com/bytefish/facerec/blob/master/py/facerec/feature.py)) and the classifier (which must be an [AbstractClassifier](https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py)) form a [PredictableModel](https://github.com/bytefish/facerec/blob/master/py/facerec/model.py), which does the feature extraction and learns the classifier.

So if you want to use the Fisherfaces method for feature extraction you would do:

```
from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel

model = PredictableModel(Fisherfaces(), NearestNeighbor())
```

Once you have created your model you can call `compute(data,labels)` to learn it on given image `data` and their labels. There's nothing like a Dataset structure I enforce: You pass the images as a list of NumPy arrays (or something that could be converted into NumPy arrays), the labels are again a NumPy arrays of integer numbers (corresponding to a person).

```python

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(self.sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]
```

Reading in the image data is then as easy as calling:
```python
# Read in the image data:
[X,y] = read_images("/path/to/your/image/data")
```
#### Learning a Model

You can then learn a model by calling `compute` on it. You have to pass the image data in a list `X` and the according labels in a list `y`:

```
# Then compute the model:
model.compute(X,y)
# ...
```

#### Getting a prediction

Since I can't assume a standard classifier output, a classifier always outputs a list with:

```
[ predicted_label, generic_classifier_output]
```

Take the k-Nearest Neighbor for example. Imagine I have a 3-Nearest Neighbor classifier, then your PredictableModel is going to return something similar to:

```
>>> model.predict(X)
[ 0, 
  { 'labels'    : [ 0,      0,      1      ],
    'distances' : [ 10.132, 10.341, 13.314 ]
  }
]
```

In this example the predicted label is `0`, because two of three nearest neighbors were of label `0` and only one neighbor was `1`. The generic output is given in a dict for these classifiers, so you are given some semantic information. I prefer this over plain Python lists, because it is probably hard to read through some code, if you are accessing stuff by indices only.

If you only want to know the predicted label for a query image `X` you would write:

```
predicted_label = model.predict(X)[0]
```

And if you want to make your `PredictableModel` more sophisticated, by rejecting examples based on the classifier output for example, then you'll need to access the generic classifier output:

```
prediction = model.predict(X)
predicted_label = prediction[0]
generic_classifier_output = prediction[1]
```

You have to read up the classifier output in the help section of each classifers predict method.

#### Thresholding

In OpenCV you can pass a decision threshold to the predict method, which a prediction is thresholded against. So how can you introduce a decision threshold in the facerec framework? I admit there isn't a convenient or obvious way to do so, but it's actually quite easy. Imagine your classifier is 1-Nearest Neighbor, then a prediciton is going to yield something like this:

```
>>> prediction = model.predict(X)
[ 0,
   { 'labels'    : [ 0 ],
     'distances' : [ 12.345 ]
   }
]
```

Where

```
prediction[0]    -- Is the predicted label.
predicition[1]   -- is the generic classifier output, the decision is based on.
```

Now let's say you have estimated, that every distance above `10.1` is nonsense and should be ignored. Then you could do something like this in your script, to threshold against the given value:

```python
# This gets you the output:
prediction = model.predict(X)
predicted_label = prediction[0]
classifier_output = prediction[1]
# Now let's get the distance from the assuming a 1-Nearest Neighbor.
# Since it's a 1-Nearest Neighbor only look take the zero-th element:
distance = classifier_output['distances'][0]
# Now you can easily threshold by it:
if distance > 10.0:
  print "Unknown Person!"
else
  print "Person is known with label %i" % (predicted_label) 
```

#### Image processing chains

Sometimes it's also necessary to perform preprocessing on your images. This framework is quite advanced and makes it easy to experiment with algorithms. You can achieve image processing chains by using the [ChainOperator](https://github.com/bytefish/facerec/blob/master/py/facerec/operators.py). The ChainOperator computes a `feature1` and passes its output to a `feature2`. See the implementation of the `ChainOperator`, which is a `FeatureOperator`. The `FeatureOperator` in turn is an `AbstractFeature` again, so it can be the input for another `AbstractFeature`. Get it?

```python

class FeatureOperator(AbstractFeature):
	"""
	A FeatureOperator operates on two feature models.
	
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
	"""
	def __init__(self,model1,model2):
		if (not isinstance(model1,AbstractFeature)) or (not isinstance(model2,AbstractFeature)):
			raise Exception("A FeatureOperator only works on classes implementing an AbstractFeature!")
		self.model1 = model1
		self.model2 = model2
	
	def __repr__(self):
		return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
		
class ChainOperator(FeatureOperator):
	"""
	The ChainOperator chains two feature extraction modules:
		model2.compute(model1.compute(X,y),y)
	Where X can be generic input data.
	
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
	"""
	def __init__(self,model1,model2):
		FeatureOperator.__init__(self,model1,model2)
		
	def compute(self,X,y):
		X = self.model1.compute(X,y)
		return self.model2.compute(X,y)
		
	def extract(self,X):
		X = self.model1.extract(X)
		return self.model2.extract(X)
	
	def __repr__(self):
		return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
```

So imagine we want to perform a TanTriggs preprocessing, before applying the Fisherfaces. What you would do is using the output of the `TanTriggsPreprocessing` feature extraction as input for the `Fisherfaces` feature extraction. We can express this with a `ChainOperator` in facerec:

```
from facerec.preprocessing import TanTriggsPreprocessing
from facerec.feature import Fisherfaces
from facerec.operators import ChainOperator
from facerec.model import PredictableModel


feature = ChainOperator(TanTriggsPreprocessing(), Fisherfaces())
classifier = NearestNeighbor()
model = PredictableModel(feature, classifier)
```

#### Classifier

Start with a [Nearest Neighbor model](https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py) as a classifier. In its simplest form you would just need to write:

```
from facerec.classifier import NearestNeighbor

classifier = NearestNeighbor()
```

This creates a 1-Nearest Neighbor with the Euclidean Distance as distance metric. To create a 5-Nearest Neighbor with a Cosine Distance instead, you would write:

```
from facerec.classifier import NearestNeighbor
from facerec.distance import CosineDistance

classifier = NearestNeighbor(dist_metric=CosineDistance(), k=5)
```

If you want to build a `PredictableModel` to generate predictions, simply use the [PredictableModel](https://github.com/bytefish/facerec/blob/master/py/facerec/model.py):

```
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from.facerec.classifier import NearestNeighbor
from facerec.distance import EuclideanDistance

feature = Fisherfaces()

classifier = NearestNeighbor(dist_metric=CosineDistance(), k=5)
predictor = PredictableModel(feature, classifier)
```

Once you have created your model you can call `compute` to learn it. Please see the [examples](https://github.com/bytefish/facerec/tree/master/py/apps/).

### sample application ###

I've hacked up the [videofacerec](https://github.com/bytefish/facerec/tree/master/py/apps/videofacerec) application, which shows how easy it is to interface with the OpenCV2 API: [py/apps/videofacerec](https://github.com/bytefish/facerec/tree/master/py/apps/videofacerec).

Here's a screenshot of the running application:

![videofacerec](https://github.com/bytefish/facerec/raw/master/py/apps/videofacerec/app_screenshot.jpg "videofacerec")

## facerec/m ##

GNU Octave implementation of parts of the Python version.

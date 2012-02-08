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

### documentation ###

I need some more time to write a detailed documentation of the framework, but there are some examples how it can be used. Please have a look at [py/apps/scripts/fisherfaces_example.py](https://github.com/bytefish/facerec/blob/master/py/apps/scripts/fisherfaces_example.py) to see how one learns a Fisherfaces model, performs a 10-fold Cross Validation and plots the Fisherfaces. Please see [bytefish.de/blog/fisherfaces](http://www.bytefish.de/blog/fisherfaces) to learn how to preprocess the images.

Basically all face recognition algorithms you build consist of 
  - [Feature Extraction](https://github.com/bytefish/facerec/blob/master/py/facerec/feature.py)
  - [Classifier](https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py)

The feature and the classifier form a [PredictableModel](https://github.com/bytefish/facerec/blob/master/py/facerec/model.py), which does the feature extraction and learns the classifier.

If you want to use the Fisherfaces method for feature extraction you would do:

```
from facerec.feature import Fisherfaces

...

feature = Fisherfaces()
```

But sometimes it's necessary to perform preprocessing on your images. You can achieve preprocessing chains by using the [ChainOperator](https://github.com/bytefish/facerec/blob/master/py/facerec/operators.py). The ChainOperator computes a feature1 and passes its output to a feature2. In the following I do a TanTriggs preprocessing before learning the Fisherfaces.

```
from facerec.preprocessing import TanTriggsPreprocessing
from facerec.feature import Fisherfaces
from facerec.operators import ChainOperator

...

feature = ChainOperator(TanTriggsPreprocessing(), Fisherfaces())
```

Start with a [Nearest Neighbor model](https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py) as a classifier. In its simplest form you just need to write:

```
from facerec.classifier import NearestNeighbor
...
classifier = NearestNeighbor()
```

Which creates a 1-Nearest Neighbor with the Euclidean Distance as metric. To create a 5-Nearest Neighbor with a Cosine Distance you would write:

```
from facerec.classifier import NearestNeighbor
from facerec.distance import CosineDistance

...

classifier = NearestNeighbor(dist_metric=CosineDistance(), k=5)
```

To build a model which can be computed and generates prediction, simply use the [PredictableModel](https://github.com/bytefish/facerec/blob/master/py/facerec/model.py):

```

from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from.facerec.classifier import NearestNeighbor
from facerec.distance import EuclideanDistance
...
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

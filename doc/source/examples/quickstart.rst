Quickstart
==========

In this chapter we will write a script to perform face recognition from a set of images. We assume, 
that the images are given in folders, which we will read and learn a face recognition model with them.

You can obtain the code in this section from:

* `facerec/py/apps/scripts/simple_example.py <https://github.com/bytefish/facerec/blob/master/py/apps/scripts/simple_example.py>`_

Getting the data right
**********************

We aren't doing a toy example, so you'll need some image data. For sake of simplicity I have assumed, that the images (the faces, persons you want to recognize) are given in folders. So imagine I have a folder ``images`` (the dataset!), with the subfolders person1, person2 and so on:

::
    
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

One of the public available datasets, that is already coming in such a folder structure is the AT&T Facedatabase, available at:

* `http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_

Once unpacked it is going to look like this (on my filesystem it is unpacked to ``/home/philipp/facerec/data/at/``, your path is different!):

::
    
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

That's all that needs to be done.

What is a PredictableModel?
***************************

Basically all face recognition algorithms are the combination of a `feature extraction <https://github.com/bytefish/facerec/blob/master/py/facerec/feature.py>`_
and a `classifier <https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py>`_. The Eigenfaces method for example is a Principal Component Analysis 
with a Nearest Neighbor classifier. The feature (which must be an `AbstractFeature <https://github.com/bytefish/facerec/blob/master/py/facerec/feature.py>`_) and 
the classifier (which must be an `AbstractClassifier <https://github.com/bytefish/facerec/blob/master/py/facerec/classifier.py>`_) form a 
`PredictableModel <https://github.com/bytefish/facerec/blob/master/py/facerec/model.py>`_), which does the feature extraction and trains the classifier.

So! If you want to use the `Fisherfaces method <http://bytefish.de/blog/fisherfaces/>`_ for feature extraction you would do:

.. code-block:: python

    from facerec.feature import Fisherfaces
    from facerec.classifier import NearestNeighbor
    from facerec.model import PredictableModel

    model = PredictableModel(Fisherfaces(), NearestNeighbor())

Once you have created your model you can call `compute(data,labels)` to learn it on given image `data` and their labels. There's nothing like a Dataset structure I enforce: You pass the images as a list of NumPy arrays (or something that could be converted into NumPy arrays), the labels are again a NumPy arrays of integer numbers (corresponding to a person).

.. code-block:: python

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

Reading in the image data is then as easy as calling:

.. code-block:: python

    # Read in the image data:
    [X,y] = read_images("/path/to/your/image/data")


The full example
****************

.. code-block:: python

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

Results
*******

Since the AT&T Facedatabase is a fairly easy database we have got a `95.5%` recognition rate with the Fisherfaces method (with a 10-fold cross validation):

::

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

And we can have a look at the Fisherfaces found by the model:

.. image:: images/fisherfaces_at.png
    :align: center
    :alt: alternate text
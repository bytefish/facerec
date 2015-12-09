#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import sys, os
sys.path.append("../..")
# facerec
from facerec.feature import Fisherfaces, PCA, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# required libraries
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

class FileNameFilter:
    def __init__(self, name):
        self._name = name

    def __call__(self, filename):
        return True
        
    def __repr__(self):
        return "FileNameFilter (name=%s)" % (self._name) 


class YaleBaseFilter(FileNameFilter):
    def __init__(self, min_azimuth, max_azimuth, min_elevation, max_elevation):
        FileNameFilter.__init__(self, "Filter YaleFDB Subset1")
        self._min_azimuth = min_azimuth
        self._max_azimuth = max_azimuth
        self._min_elevation = min_elevation
        self._max_elevation = max_elevation

    def __call__(self, filename):
        # We only want the PGM files:
        
        filetype = filename[-4:]
        if filetype != ".pgm":
            return False

        # There are "Ambient" PGM files, ignore them:
        if "Ambient" in filename:
            return False
        
        azimuth = int(filename[12:16])
        elevation = int(filename[17:20])

        # Now filter based on angles:
        if azimuth < self._min_azimuth or azimuth > self._max_azimuth:
            return False
        if elevation < self._min_elevation or elevation > self._max_elevation:
            return False
            
        return True

def read_images(path, fileNameFilter=FileNameFilter("None"), sz=None):
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
                if fileNameFilter(filename):
                    print filename
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
    yale_filter = YaleBaseFilter(-25, 25, -25, 25)
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1], yale_filter)
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = PCA()
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
    cv.print_results()

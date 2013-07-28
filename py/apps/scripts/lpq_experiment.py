#!/usr/bin/python
#
# coding: utf-8
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Philipp Wagner <bytefish[at]gmx[dot]de>.
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

import numpy as np
from scipy import ndimage
import os
import sys

sys.path.append("../..")

from PIL import Image
import matplotlib.pyplot as plt

import logging

from facerec.feature import PCA, Fisherfaces, SpatialHistogram
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.lbp import LPQ, ExtendedLBP
from facerec.validation import KFoldCrossValidation, ValidationResult, precision


EXPERIMENT_NAME = "LocalPhaseQuantizationExperiment"

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
    
def apply_gaussian(X, sigma):
    """A simple function to apply a Gaussian Blur on each image in X.
    
    Args:
        X: A list of images.
        sigma: sigma to apply
        
    Returns:
        Y: The processed images
    """
    return np.array([ndimage.gaussian_filter(x, sigma) for x in X])


def results_to_list(validation_results):
    return [precision(result.true_positives,result.false_positives) for result in validation_results]
    
    
if __name__ == "__main__":
    # This is where we write the results to, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: lpq_experiment.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1])
    # Set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # The models we want to evaluate:
    model0 = PredictableModel(feature=PCA(num_components=50), classifier=NearestNeighbor(dist_metric=EuclideanDistance(), k=1))
    model1 = PredictableModel(feature=Fisherfaces(), classifier=NearestNeighbor(dist_metric=EuclideanDistance(), k=1))
    model2 = PredictableModel(feature=SpatialHistogram(lbp_operator=ExtendedLBP()), classifier=NearestNeighbor(dist_metric=ChiSquareDistance(), k=1))
    model3 = PredictableModel(feature=SpatialHistogram(lbp_operator=LPQ()), classifier=NearestNeighbor(dist_metric=ChiSquareDistance(), k=1))
    # I should rewrite the framework to offer a less memory-intense solution here:
    cv0 = KFoldCrossValidation(model0, k=10)
    cv1 = KFoldCrossValidation(model1, k=10)
    cv2 = KFoldCrossValidation(model2, k=10)
    cv3 = KFoldCrossValidation(model3, k=10)
    # Make it a list, so we can iterate through:
    validators = [cv0, cv1, cv2, cv3]
    # The sigmas we'll apply for each run:
    sigmas = [0, 1, 2, 4]
    # If everything went fine, we should have the results of each model:
    for sigma in sigmas:
        Xs = apply_gaussian(X, sigma)
        for validator in validators:
            experiment_description = "%s (sigma=%.2f)" % (EXPERIMENT_NAME, sigma)
            validator.validate(Xs, y, experiment_description)
    # Print the results:
    for validator in validators:
        validator.print_results()
    # Make a nice plot of this textual output:
    fig = plt.figure()
    fig.text(.5, .95, EXPERIMENT_NAME, horizontalalignment='center') 
    # Add the Validation results:
    plt.plot(
        sigmas, results_to_list(cv0.validation_results), 'r--', 
        sigmas, results_to_list(cv1.validation_results), 'bs',
        sigmas, results_to_list(cv2.validation_results), 'g^',
        sigmas, results_to_list(cv3.validation_results), 'k')
    # Add a Legend:
    plt.legend((cv0, cv1, cv2, cv3), 'lower right', shadow=True, fancybox=True)
    # Scale Precision correctly:
    plt.ylim(0,1)
    # Finally add the labels:
    plt.ylabel('Precision')
    plt.xlabel('Sigma')
    # Save the gifure and we are out of here!
    fig.savefig("lpq_experiment.png")
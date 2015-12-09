#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
from scipy import ndimage
import os
import sys

sys.path.append("../..")

# try to import the PIL Image
try:
    from PIL import Image
except ImportError:
    import Image

import matplotlib.pyplot as plt
import textwrap

import logging

from facerec.feature import SpatialHistogram
from facerec.distance import ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.lbp import LPQ, ExtendedLBP
from facerec.validation import SimpleValidation, precision
from facerec.util import shuffle_array

EXPERIMENT_NAME = "LocalPhaseQuantizationExperiment"

# ITER_MAX is the number of experimental runs, as described in the 
# original paper. For testing purposes, it was set to 1, but it
# should be set to a higher value to get at least a little confidence
# in the results.
ITER_MAX = 1

class FileNameFilter:
    """
    Base class used for filtering files.
    """
    def __init__(self, name):
        self._name = name

    def __call__(self, filename):
        return True
        
    def __repr__(self):
        return "FileNameFilter (name=%s)" % (self._name) 


class YaleBaseFilter(FileNameFilter):
    """
    This Filter filters files, based on their filetype ending (.pgm) and
    their azimuth and elevation. The higher the angle, the more shadows in
    the face. This is useful for experiments with illumination and 
    preprocessing. 
    
    """
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
        
        azimuth = abs(int(filename[12:16]))
        elevation = abs(int(filename[17:20]))

        # Now filter based on angles:
        if azimuth < self._min_azimuth or azimuth > self._max_azimuth:
            return False
        if elevation < self._min_elevation or elevation > self._max_elevation:
            return False
            
        return True

    def __repr__(self):
        return "Yale FDB Filter (min_azimuth=%s, max_azimuth=%s, min_elevation=%s, max_elevation=%s)" % (min_azimuth, max_azimuth, min_elevation, max_elevation)


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
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        im = im.convert("L")
                        # resize to given size (if given)
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
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
    
def partition_data(X, y):
    """
    Shuffles the input data and splits it into a new set of images. This resembles the experimental setup
    used in the paper on the Local Phase Quantization descriptor in:
    
        "Recognition of Blurred Faces Using Local Phase Quantization", Timo Ahonen, Esa Rahtu, Ville Ojansivu, Janne Heikkila

    What it does is to build a subset for each class, so it has 1 image for training and the rest for testing. 
    The original dataset is shuffled for each call, hence you always get a new partitioning.

    """
    Xs,ys = shuffle_array(X,y)
    # Maps index to class:
    mapping = {}
    for i in xrange(len(y)):
        yi = ys[i]
        try:
            mapping[yi].append(i)
        except KeyError:
            mapping[yi] = [i]
    # Get one image for each subject:
    Xtrain, ytrain = [], []
    Xtest, ytest = [], []
    # Finally build partition:
    for key, indices in mapping.iteritems():
        # Add images:
        Xtrain.extend([ Xs[i] for i in indices[:1] ])
        ytrain.extend([ ys[i] for i in indices[:1] ])
        Xtest.extend([ Xs[i] for i in indices[1:20]])
        ytest.extend([ ys[i] for i in indices[1:20]])
    # Return shuffled partitions:
    return Xtrain, ytrain, Xtest, ytest

class ModelWrapper:
    def __init__(model):
        self.model = model
        self.result = []

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
    # Define filters for the Dataset:
    yale_subset_0_40 = YaleBaseFilter(0, 40, 0, 40)
    # Now read in the image data. Apply filters, scale to 128 x 128 pixel:
    [X,y] = read_images(sys.argv[1], yale_subset_0_40, sz=(64,64))
    # Set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # The models we want to evaluate:
    model0 = PredictableModel(feature=SpatialHistogram(lbp_operator=ExtendedLBP()), classifier=NearestNeighbor(dist_metric=ChiSquareDistance(), k=1))
    model1 = PredictableModel(feature=SpatialHistogram(lbp_operator=LPQ()), classifier=NearestNeighbor(dist_metric=ChiSquareDistance(), k=1))
    # The sigmas we'll apply for each run:
    sigmas = [0]
    print 'The experiment will be run %s times!' % ITER_MAX
    # Initialize experiments (with empty results):
    experiments = {}
    experiments['lbp_model'] = { 'model': model0, 'results' : {}, 'color' : 'r', 'linestyle' : '--', 'marker' : '*'} 
    experiments['lpq_model'] = { 'model': model1, 'results' : {}, 'color' : 'b', 'linestyle' : '--', 'marker' : 's'}
    # Loop to acquire the results for each experiment:
    for sigma in sigmas:
        print "Setting sigma=%s" % sigma
        for key, value in experiments.iteritems():
            print 'Running experiment for model=%s' % key
            # Define the validators for the model:
            cv0 = SimpleValidation(value['model'])
            for iteration in xrange(ITER_MAX):
                print "Repeating experiment %s/%s." % (iteration + 1, ITER_MAX)
                # Split dataset according to the papers description:
                Xtrain, ytrain, Xtest, ytest = partition_data(X,y)
                # Apply a gaussian blur on the images:
                Xs = apply_gaussian(Xtest, sigma)
                # Run each validator with the given data:
                experiment_description = "%s (iteration=%s, sigma=%.2f)" % (EXPERIMENT_NAME, iteration, sigma)
                cv0.validate(Xtrain, ytrain, Xs, ytest, experiment_description)
            # Get overall results:
            true_positives = sum([validation_result.true_positives for validation_result in cv0.validation_results])
            false_positives = sum([validation_result.false_positives for validation_result in cv0.validation_results])
            # Calculate overall precision:
            prec = precision(true_positives,false_positives)
            # Store the result:
            print key
            experiments[key]['results'][sigma] = prec

    # Make a nice plot of this textual output:
    fig = plt.figure()
    # Holds the legend items:
    plot_legend = []
    # Add the Validation results:
    for experiment_name, experiment_definition in experiments.iteritems():
        print key, experiment_definition
        results = experiment_definition['results']
        (xvalues, yvalues) = zip(*[(k,v) for k,v in results.iteritems()])
        # Add to the legend:
        plot_legend.append(experiment_name)
        # Put the results into the plot:
        plot_color = experiment_definition['color']
        plot_linestyle = experiment_definition['linestyle']
        plot_marker = experiment_definition['marker']
        plt.plot(sigmas, yvalues, linestyle=plot_linestyle, marker=plot_marker, color=plot_color)
    # Put the legend below the plot (TODO):
    plt.legend(plot_legend, prop={'size':6}, numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.2),  fancybox=True, shadow=True, ncol=1)
    # Scale y-axis between 0,1 to see the Precision:
    plt.ylim(0,1)
    plt.xlim(-0.2, max(sigmas) + 1)
    # Finally add the labels:
    plt.title(EXPERIMENT_NAME)
    plt.ylabel('Precision')
    plt.xlabel('Sigma')
    fig.subplots_adjust(bottom=0.5)
    # Save the gifure and we are out of here!
    plt.savefig("lpq_experiment.png", bbox_inches='tight',dpi=100)

#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2014, Philipp Wagner <bytefish[at]gmx[dot]de>.
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

import csv, os, sys
# Import PIL
try:
    from PIL import Image
except ImportError:
    import Image
# Import numpy:
import numpy as np
# Import facerec:
from facerec.feature import ChainOperator, Fisherfaces
from facerec.preprocessing import Resize
from facerec.dataset import NumericDataSet
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model

# This is the face recognition module for the RESTful Webservice.
#
# The current implementation uses a fixed model defined in code. 
# A simple wrapper works around a limitation of the current framework, 
# because as time of writing this only integer labels can be passed into 
# the classifiers of the facerec framework. 

# This wrapper hides the complexity of dealing with integer labels for
# the training labels. It also supports updating a model, instead of 
# re-training it. 
class PredictableModelWrapper(object):

    def __init__(self, model):
        self.model = model
        self.numeric_dataset = NumericDataSet()
        
    def compute(self):
        X,y = self.numeric_dataset.get()
        self.model.compute(X,y)

    def set_data(self, numeric_dataset):
        self.numeric_dataset = numeric_dataset

    def predict(self, image):
        prediction_result = self.model.predict(image)
        # Only take label right now:
        num_label = prediction_result[0]
        str_label = self.numeric_dataset.resolve_by_num(num_label)
        return str_label

    def update(self, name, image):
        self.numeric_dataset.add(name, image)
        class_label = self.numeric_dataset.resolve_by_str(name)
        extracted_feature = self.feature.extract(image)
        self.classifier.update(extracted_feature, class_label)

    def __repr__(self):
        return "PredictableModelWrapper (Inner Model=%s)" % (str(self.model))


# Now define a method to get a model trained on a NumericDataSet,
# which should also store the model into a file if filename is given.
def get_model(numeric_dataset, model_filename=None):
    feature = ChainOperator(Resize((128,128)), Fisherfaces())
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    inner_model = PredictableModel(feature=feature, classifier=classifier)
    model = PredictableModelWrapper(inner_model)
    model.set_data(numeric_dataset)
    model.compute()
    if not model_filename is None:
        save_model(model_filename, model)
    return model

# Now a method to read images from a folder. It's pretty simple,
# since we can pass a numeric_dataset into the read_images  method 
# and just add the files as we read them. 
def read_images(path, identifier, numeric_dataset):
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, filename))
            img = img.convert("L")
            img = np.asarray(img, dtype=np.uint8)
            numeric_dataset.add(identifier, img)
        except IOError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

# read_csv is a tiny little method, that takes a csv file defined
# like this:
#
#   Philipp Wagner;D:/images/philipp
#   Another Name;D:/images/another_name
#   ...
#
def read_from_csv(filename):
    numeric_dataset = NumericDataSet()
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='#')
        for row in reader:
            identifier = row[0]
            path = row[1]
            read_images(path, identifier, numeric_dataset)
    return numeric_dataset

# Just some sugar on top...
def get_model_from_csv(filename, out_model_filename):
    numeric_dataset = read_from_csv(filename)
    model = get_model(numeric_dataset, out_model_filename)
    return model

def load_model_file(model_filename):
    load_model(model_filename)

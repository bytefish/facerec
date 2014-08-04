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

import sys
# append facerec
sys.path.append("../..")

from facerec.feature import ChainOperator, Fisherfaces
from facerec.preprocessing import Resize
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
#
#
#
# First of all define a model, that supports updating itself. This 
# is necessary, so we don't need to retrain the entire model for each 
# input image. This is not suitable for all models, it may be limited 
# to Local Binary Patterns for the current framework.
#
# Moreover it hides the complexity of dealing with integer labels for
# the faces (or objects) to identify.
#
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
        class_label = self.model.predict(image)
        return self.numeric_dataset.resolve_by_num(class_label)

    def update(self, name, image):
        self.numeric_dataset.add(name, image)
        class_label = self.numeric_dataset.resolve_by_str(name)
        extracted_feature = self.feature.extract(image)
        self.classifier.update(extracted_feature, class_label)
        
# To abstract the dirty things away, we are going to use a 
# new class, which we call a NumericDataSet. This NumericDataSet
# allows us to add images and turn them into a facerec compatible
# representation.
class NumericDataSet(object):
    def __init__(self):
        self.data = {}
        self.str_to_num_mapping = {}
        self.num_to_str_mapping = {}

    def add(self, identifier, image):
        numeric_identifier = self.__resolve_subject_id(identifier)
        try:
            self.data[identifier].append(image)
        except:
            self.data[identifier] = [image]
            numerical_identifier = len(self.str_to_num_mapping)
            # Store in mapping tables:
            self.str_to_num_mapping[identifier] = numerical_identifier
            self.num_to_str_mapping[numerical_identifier] = identifier

    def get(self):
        X = []
        y = []
        for name, num in self.str_to_num_mapping:
            for image in self.data[name]:
                X.append(image)
                y.append(num)
        return X,y

    def resolve_by_str(self, identifier):
        return self.str_num_mapping[identifier]

    def resolve_by_num(self, numerical_identifier):
        return self.num_to_str_mapping[numerical_identifier]

    def length(self):
        return len(self.data)

    def __repr__(self):
        print "TODO"

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
import cStringIO
import base64
# try to import the PIL Image
try:
    from PIL import Image
except ImportError:
    import Image
# Flask imports:
from flask import Flask, request, json
# facerec imports:
import sys
sys.path.append("../..")
# facerec imports
from facerec.model import PredictableModel
from facerec.lbp import ExtendedLBP
from facerec.feature import SpatialHistogram
from facerec.distance import ChiSquareDistance
from facerec.classifier import NearestNeighbor
# logging
import logging

# Test run:
# Encode image as Base64:
#   openssl enc -base64 -in D:\facerec\data\c1\crop_arnold_schwarzenegger\crop_01.jpg
# Send request (Linux/Cygwin):
#   curl -i -H "Content-Type: application/json" -X POST -d '{"name":"Arnie", "image":""}' http://localhost:5000/add
# Send request (Windows):
# curl -i -H "Content-Type: application/json" -X POST -d "{"""name""":"""Arnie""", """image""":"""base64image"""}" http://localhost:5000/add

# The main application: 
app = Flask(__name__)

# Limit the maximum image size:
IMG_MAX_SIZE = (128,128)

# Define a model, that supports updating itself. This is necessary,
# so we don't need to retrain the entire model for each input image.
# This is not suitable for all models, it may be limited to Local 
# Binary Patterns for the current framework:
class WebAppException(Exception):
    pass
    
class UpdatableModel(PredictableModel):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """

    def __init__(self, feature, classifier):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.subject_names = []

    def update(self, image, name):
        c = self.__resolve_subject_id(name)
        Y = self.feature.extract(image)
        self.classifier.update(Y,c)

    def predict_image(self, image):
        y = self.predict(image)
        return self.__resolve_subject_name(y[0])
        
    def __resolve_subject_id(self, query_name):
        # Do we have it in the list?
        for pos in xrange(len(self.subject_names)):
            if self.subject_names[pos] == query_name:
                return pos
        # If not, add it!
        self.subject_names.append(query_name)
        return len(self.subject_names) - 1

    def __resolve_subject_name(self, query_id):
        if len(self.subject_names) == 0:
            raise WebAppException("No subjects available!")
        return self.subject_names[query_id]

def read_image(base64_image):
    enc_data = base64.b64decode(base64_image)
    file_like = cStringIO.StringIO(enc_data)
    im = Image.open(file_like)
    return im.convert("L")

def resize_image(image, sz):
    """ Resizes an image, so it doesn't exceed the maximum size given in sz. This function
    sucks, because it doesn't respect any image ration. Just ignore it for this prototype.
    """
    (width, height) = image.size
    if width > sz[0] or height > sz[1]:
        new_width = min(sz[0], width)
        new_height = min(sz[1], height)
        image = image.resize(sz, Image.ANTIALIAS)
    return image

model = UpdatableModel(feature=SpatialHistogram(lbp_operator=ExtendedLBP()), classifier=NearestNeighbor(dist_metric=ChiSquareDistance(), k=1))

@app.route('/add', methods=["POST", "GET"])
def add():
    print "add"
    if request.headers['Content-Type'] == 'application/json':
        values = request.json
        # Read the image:
        image = read_image(values['image'])
        image = resize_image(image, IMG_MAX_SIZE)
        # And update the model:
        subject_name = values['name']
        print "Model update."
        model.update(image,subject_name)
        
@app.route('/predict', methods=["POST"])
def predict():
    print "predict"
    if request.headers['Content-Type'] == 'application/json':
        values = request.json
        # Read the image:
        image = read_image(values['image'])
        image = resize_image(image, IMG_MAX_SIZE)
        # Get the predicted name
        return model.predict_image(image)

if __name__ == '__main__':

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    app.run(host="0.0.0.0", port=int("5000"), debug=True)

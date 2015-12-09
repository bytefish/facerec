#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import cStringIO
import base64
# try to import the PIL Image
try:
    from PIL import Image
except ImportError:
    import Image

# Flask imports:
from flask import Flask, request, request_finished, json, abort, make_response, Response, jsonify
# facerec imports
# facerec imports:
import sys
sys.path.append("../../..")
from facerec.model import PredictableModel
from facerec.lbp import ExtendedLBP
from facerec.feature import SpatialHistogram
from facerec.distance import ChiSquareDistance
from facerec.classifier import NearestNeighbor

# logging
import logging
from logging.handlers import RotatingFileHandler

# the webserver recognition module
import recognition

# The main application: 
app = Flask(__name__)

# This is a list of errors the Webservice returns. You can come up
# with new error codes and plug them into the API.
#
# An example JSON response for an error looks like this:
#
#   { "status" : failed, "message" : "IMAGE_DECODE_ERROR", "code" : 10 }
#
# If there are multiple errors, only the first error is considered.

IMAGE_DECODE_ERROR = 10
IMAGE_RESIZE_ERROR = 11
PREDICTION_ERROR = 12
SERVICE_TEMPORARY_UNAVAILABLE = 20
UNKNOWN_ERROR = 21
INVALID_FORMAT = 30
INVALID_API_KEY = 31
INVALID_API_TOKEN = 32
MISSING_ARGUMENTS = 40

errors = {
    IMAGE_DECODE_ERROR : "IMAGE_DECODE_ERROR",
    IMAGE_RESIZE_ERROR  : "IMAGE_RESIZE_ERROR",
    SERVICE_TEMPORARY_UNAVAILABLE	: "SERVICE_TEMPORARILY_UNAVAILABLE",
    PREDICTION_ERROR : "PREDICTION_ERROR",
    UNKNOWN_ERROR : "UNKNOWN_ERROR",
    INVALID_FORMAT : "INVALID_FORMAT",
    INVALID_API_KEY : "INVALID_API_KEY",
    INVALID_API_TOKEN : "INVALID_API_TOKEN",
    MISSING_ARGUMENTS : "MISSING_ARGUMENTS"
}

# Setup the logging for the server, so we can log all exceptions
# away. We also want to acquire a logger for the facerec framework,
# so we can be sure, that all logging goes into one place.
LOG_FILENAME = 'serverlog.log'
LOG_BACKUP_COUNT = 5
LOG_FILE_SIZE_BYTES = 50 * 1024 * 1024

def init_logger(app):
    handler = RotatingFileHandler(LOG_FILENAME, maxBytes=LOG_FILE_SIZE_BYTES, backupCount=LOG_BACKUP_COUNT)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    loggers = [app.logger, logging.getLogger('facerec')]
    for logger in loggers:
        logger.addHandler(handler)

# Bring the model variable into global scope. This might be
# dangerous in Flask, I am trying to figure out, which is the
# best practice here. 

# Initializes the Flask application, which is going to 
# add the loggers, load the initial facerec model and 
# all of this.
def init_app(app):
    init_logger(app)

init_app(app)

@app.before_request
def log_request():
    app.logger.debug("Request: %s %s", request.method, request.url)
    
# The WebAppException might be useful. It enables us to 
# throw exceptions at any place in the application and give the user
# a custom error code.
class WebAppException(Exception):

    def __init__(self, error_code, exception, status_code=None):
        Exception.__init__(self)
        self.status_code = 400
        self.exception = exception
        self.error_code = error_code
        try:
            self.message = errors[self.error_code]
        except:
            self.error_code = UNKNOWN_ERROR
            self.message = errors[self.error_code]
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict()
        rv['status'] = 'failed'
        rv['code'] = self.error_code
        rv['message'] = self.message
        return rv

# Wow, a decorator! This enables us to catch Exceptions 
# in a method and raise a new WebAppException with the 
# original Exception included. This is a quick and dirty way
# to minimize error handling code in our server.
class ThrowsWebAppException(object):
   def __init__(self, error_code, status_code=None):
      self.error_code = error_code
      self.status_code = status_code

   def __call__(self, function):
      def returnfunction(*args, **kwargs):
         try:
            return function(*args, **kwargs)
         except Exception as e:
            raise WebAppException(self.error_code, e)
      return returnfunction

# Register an error handler on the WebAppException, so we
# can return the error as JSON back to the User. At the same
# time you should do some logging, so it doesn't pass by 
# silently.
@app.errorhandler(WebAppException)
def handle_exception(error):
    app.logger.exception(error.exception)
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# Now finally add the methods needed for our FaceRecognition API!
# Right now there is no rate limiting, no auth tokens and so on.
# 
@ThrowsWebAppException(error_code = IMAGE_DECODE_ERROR)
def read_image(base64_image):
    """ Decodes Base64 image data, reads it with PIL and converts it into grayscale.

    Args:
    
        base64_image [string] A Base64 encoded image (all types PIL supports).
    """
    enc_data = base64.b64decode(base64_image)
    file_like = cStringIO.StringIO(enc_data)
    im = Image.open(file_like)
    im = im.convert("L")
    return im

def preprocess_image(image_data):
    image = read_image(image_data)
    return image

# Get the prediction from the global model.
@ThrowsWebAppException(error_code = PREDICTION_ERROR)
def get_prediction(image_data):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    return prediction

# Now add the API endpoints for recognizing, learning and 
# so on. If you want to use this in any public setup, you
# should add rate limiting, auth tokens and so on.
@app.route('/api/recognize', methods=['GET', 'POST'])
def identify():
    if request.headers['Content-Type'] == 'application/json':
            try:
                image_data = request.json['image']
            except:
                raise WebAppException(error_code=MISSING_ARGUMENTS)
            prediction = get_prediction(image_data)
            response = jsonify(name = prediction) 
            return response
    else:
        raise WebAppException(error_code=INVALID_FORMAT)

# And now let's do this!
if __name__ == '__main__':
    # A long description:
    long_description = ("server.py is a simple facerec webservice. It provides "
        "you with a simple RESTful API to recognize faces from a "
        "computed model. Please don't use this server in a production "
        "environment, as it provides no security and there might be "
        "ugly concurrency issues with the global state of the model." )
    print "=== Description ==="
    print long_description
    # Parse the command line:    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--train", action="store", dest="dataset", default=None, 
        help="Calculates a new model from a given CSV file. CSV format: <person>;</path/to/image/folder>.", required=False)
    parser.add_argument("-a", "--address", action="store", dest="host", default="0.0.0.0", 
        help="Sets the endpoint for this server.", required=False)
    parser.add_argument("-p", "--port", action="store", dest="port", default=5000, 
        help="Sets the port for this server.", required=False)
    parser.add_argument('model_filename', nargs='?', help="Filename of the model to use or store")
    # Print Usage:
    print "=== Usage ==="
    parser.print_help()
    # Parse the Arguments:
    args = parser.parse_args()
    # Uh, this is ugly...
    global model
    # If a DataSet is given, we want to work with it:
    if args.dataset:
        # Learn the new model with the dataset given:
        model = recognition.get_model_from_csv(filename=args.dataset,out_model_filename=args.model_filename)
    else:
        model = recognition.load_model_file(args.model_filename)
    # Finally start the server:        
    print "=== Server Log (also in %s) ===" % (LOG_FILENAME)
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False, threaded=False)

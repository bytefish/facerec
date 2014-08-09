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

# Limit the maximum image size:
IMG_MAX_SIZE = (128, 128)

# This is a list of errors the Webservice returns. You can come up
# with new error codes 
#
# An example JSON response for an error looks like this:
#
#   { "status" : failed, "message" : "IMAGE_DECODE_ERROR", "code" : 10 }
#
# If there are multiple errors, only the first error is considered.

IMAGE_DECODE_ERROR = 10
IMAGE_RESIZE_ERROR = 11
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
    loggers = [app.logger, logging.getLogger("facerec")]
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

@app.before_request
def log_request():
    app.logger.debug("Request: %s %s", flask.request.method, flask.request.url)
    
# The WebAppException might be useful. It enables us to 
# throw exceptions at any place in the application and give the user
# a 400 error code.
class WebAppException(Exception):
    status_code = 400

    def __init__(self, error_code, exception, status_code=None):
        Exception.__init__(self)
        self.exception = exception
        try:
            self.message = errors[error_code]
        except:
            self.error_code = UNKNOWN_ERROR
            self.message = errors[error_code]
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
            raise WebAppException(error_code, e)
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
def decodeBase64Image(base64_image):
    """ Decodes Base64 image data, reads it with PIL and converts it into grayscale.

    Args:
    
        base64_image [string] A Base64 encoded image (all types PIL supports).
    """
    enc_data = base64.b64decode(base64_image)
    file_like = cStringIO.StringIO(enc_data)
    im = Image.open(file_like)
    im = im.convert("L")
    return im

@ThrowsWebAppException(error_code = IMAGE_RESIZE_ERROR )
def resize_image(image, max_width, max_height):
    """ Resizes an image to the maximum width and height given.

    Args:

        image [image] A PIL Image object
        max_width [int] The maximum width of the result
        max_height [int] The maximum height of the result
        
    """
    (width, height) = image.size
    if width > max_width or height > max_height:
        image.thumbnail((max_width,max_height), Image.ANTIALIAS)
    return image

def resize_image(image, max_size):
    """ Resizes an image to the maximum size given.

    Args:

        image [image] A PIL Image object
        max_size [int] The maximum size
    """
    return resize_image(image, max_size[0], max_size[1])

def preprocess_image(image_data):
    image = read_image(image_data)
    image = resize_image(image, IMG_MAX_SIZE)
    return image

def get_prediction(image_data):
    image = preprocess_image(image_data)
    prediction = model.predict_image(image)
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
    usage = "usage: %prog [options] model_filename"
    print usage

    # Parse the command line:    
    from optparse import OptionParser

    parser = OptionParser(usage=usage)
    parser.add_option("-t", "--train", action="store", type="string", dest="dataset", default=None, 
        help="Calculates a new model from a given CSV file. CSV format: <person>;</path/to/image/folder>.")
     # Split between options and arguments
    (options, args) = parser.parse_args()
    # Check if a model filename was passed:
    if len(args) == 0:
        print "Expected a facerec model to use for recognition."
        sys.exit()    
    # The filename of the model:
    model_filename = args[0]
    
    # Uh, this is ugly...
    global model
    # If a DataSet is given, we want to work with it:
    if options.dataset:
        # Learn the new model with the dataset given:
        model = recognition.get_model_from_csv(filename=options.dataset,out_model_filename=model_filename)
    else:
        model = recognition.load_model_file(model_filename)
    # Finally start the server:        
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)

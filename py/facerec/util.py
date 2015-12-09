#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
import random
from scipy import ndimage

def read_image(filename):
    imarr = np.array([])
    try:
        im = Image.open(os.path.join(filename))
        im = im.convert("L") # convert to greyscale
        imarr = np.array(im, dtype=np.uint8)
    except IOError as (errno, strerror):
        print "I/O error({0}): {1}".format(errno, strerror)
    except:
        print "Cannot open image."
    return imarr

def asRowMatrix(X):
    """
    Creates a row-matrix from multi-dimensional data items in list l.
    
    X [list] List with multi-dimensional data.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([0, total], dtype=X[0].dtype)
    for row in X:
        mat = np.append(mat, row.reshape(1,-1), axis=0) # same as vstack
    return np.asmatrix(mat)

def asColumnMatrix(X):
    """
    Creates a column-matrix from multi-dimensional data items in list l.
    
    X [list] List with multi-dimensional data.
    """
    if len(X) == 0:
        return np.array([])
    total = 1
    for i in range(0, np.ndim(X[0])):
        total = total * X[0].shape[i]
    mat = np.empty([total, 0], dtype=X[0].dtype)
    for col in X:
        mat = np.append(mat, col.reshape(-1,1), axis=1) # same as hstack
    return np.asmatrix(mat)


def minmax_normalize(X, low, high, minX=None, maxX=None, dtype=np.float):
    """ min-max normalize a given matrix to given range [low,high].
    
    Args:
        X [rows x columns] input data
        low [numeric] lower bound
        high [numeric] upper bound
    """
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    minX = float(minX)
    maxX = float(maxX)
    # Normalize to [0...1].    
    X = X - minX
    X = X / (maxX - minX)
    # Scale to [low...high].
    X = X * (high-low)
    X = X + low
    return np.asarray(X, dtype=dtype)

def zscore(X):
    X = np.asanyarray(X)
    mean = X.mean()
    std = X.std() 
    X = (X-mean)/std
    return X, mean, std

def shuffle(X,y):
    idx = np.argsort([random.random() for i in xrange(y.shape[0])])
    return X[:,idx], y[idx]

def shuffle_array(X,y):
    """ Shuffles two arrays!
    """
    idx = np.argsort([random.random() for i in xrange(len(y))])
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    return (X, y)
    



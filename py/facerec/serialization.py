#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from sklearn.externals import joblib

def save_model(filename, model):
    joblib.dump(model, filename, compress=9)
    
def load_model(filename):
    return joblib.load(filename)
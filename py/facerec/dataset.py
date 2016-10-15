#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

class NumericDataSet(object):
    def __init__(self):
        self.data = {}
        self.str_to_num_mapping = {}
        self.num_to_str_mapping = {}

    def add(self, label, image):
        try:
            self.data[label].append(image)
        except:
            self.data[label] = [image]
            numerical_identifier = len(self.str_to_num_mapping)
            # Store in mapping tables:
            self.str_to_num_mapping[label] = numerical_identifier
            self.num_to_str_mapping[numerical_identifier] = label

    def get(self):
        X = []
        y = []
        for name, num in self.str_to_num_mapping.iteritems():
            for image in self.data[name]:
                X.append(image)
                y.append(num)
        return X,y

    def resolve_by_str(self, label):
        return self.str_to_num_mapping[label]

    def resolve_by_num(self, numerical_identifier):
        return self.num_to_str_mapping[numerical_identifier]

    def length(self):
        return len(self.data)

    def __repr__(self):
        print("NumericDataSet")

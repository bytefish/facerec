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


# To abstract the dirty things away, we are going to use a 
# new class, which we call a NumericDataSet. This NumericDataSet
# allows us to add images and turn them into a facerec compatible
# representation.
#
# This DataSet does not provide a method for removing entities yet,
# because this would render the ordering of the labels useless. 
# This is caused by a severe limitation of the framework right now,
# because it would make algorithms like LDA and PCA fail.
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
        print "NumericDataSet"

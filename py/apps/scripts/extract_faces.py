#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import sys
# append facerec to module search path
sys.path.append("../..")
import cv2
from facedet.detector import SkinFaceDetector
import numpy as np
import os


def extract_faces(src_dir, dst_dir, detector, face_sz = (130,130)):
	"""
	Extracts the faces from all images in a given src_dir and writes the extracted faces
	to dst_dir. Needs a facedet.Detector object to perform the actual detection.
	
	Args:
		src_dir [string] 
		dst_dir [string] 
		detector [facedet.Detector]
		face_sz [tuple] 
	"""
	if not os.path.exists(dst_dir):
		try:
			os.mkdir(dst_dir)
		except:
			raise OSError("Can't create destination directory (%s)!" % (dst_dir))
	for dirname, dirnames, filenames in os.walk(src_dir):
		for subdir in dirnames:
				src_subdir = os.path.join(dirname, subdir)
				dst_subdir = os.path.join(dst_dir,subdir)
				if not os.path.exists(dst_subdir):
					try:
						os.mkdir(dst_subdir)
					except:
						raise OSError("Can't create destination directory (%s)!" % (dst_dir))
				for filename in os.listdir(src_subdir):
					name, ext = os.path.splitext(filename)
					src_fn = os.path.join(src_subdir,filename)
					img = cv2.imread(src_fn)
					rects = detector.detect(img)
					for i,rect in enumerate(rects):
						x0,y0,x1,y1 = rect
						face = img[y0:y1,x0:x1]
						face = cv2.resize(face, face_sz, interpolation = cv2.INTER_CUBIC)
						print os.path.join(dst_subdir, "%s_%s_%d%s" % (subdir, name,i,ext))
						cv2.imwrite(os.path.join(dst_subdir, "%s_%s_%d%s" % (subdir, name,i,ext)), face)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "usage: python extract_faces.py <src_dir> <dst_dir>"
		sys.exit()
	src_dir = sys.argv[1]
	dst_dir = sys.argv[2]
	detector = SkinFaceDetector(threshold=0.3, cascade_fn="/home/philipp/projects/opencv2/OpenCV-2.3.1/data/haarcascades/haarcascade_frontalface_alt2.xml")
	extract_faces(src_dir=src_dir, dst_dir=dst_dir, detector=detector)

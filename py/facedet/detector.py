#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import sys
import os
import cv2
import numpy as np
	
class Detector:
	def detect(self, src):
		raise NotImplementedError("Every Detector must implement the detect method.")

class SkinDetector(Detector):
	"""
	Implements common color thresholding rules for the RGB, YCrCb and HSV color 
	space. The values are taken from a paper, which I can't find right now, so
	be careful with this detector.
	
	"""
	def _R1(self,BGR):
		# channels
		B = BGR[:,:,0]
		G = BGR[:,:,1]
		R = BGR[:,:,2]
		e1 = (R>95) & (G>40) & (B>20) & ((np.maximum(R,np.maximum(G,B)) - np.minimum(R, np.minimum(G,B)))>15) & (np.abs(R-G)>15) & (R>G) & (R>B)
		e2 = (R>220) & (G>210) & (B>170) & (abs(R-G)<=15) & (R>B) & (G>B)
		return (e1|e2)
	
	def _R2(self,YCrCb):
		Y = YCrCb[:,:,0]
		Cr = YCrCb[:,:,1]
		Cb = YCrCb[:,:,2]
		e1 = Cr <= (1.5862*Cb+20)
		e2 = Cr >= (0.3448*Cb+76.2069)
		e3 = Cr >= (-4.5652*Cb+234.5652)
		e4 = Cr <= (-1.15*Cb+301.75)
		e5 = Cr <= (-2.2857*Cb+432.85)
		return e1 & e2 & e3 & e4 & e5
	
	def _R3(self,HSV):
		H = HSV[:,:,0]
		S = HSV[:,:,1]
		V = HSV[:,:,2]
		return ((H<25) | (H>230))
	
	def detect(self, src):
		if np.ndim(src) < 3:
			return np.ones(src.shape, dtype=np.uint8)
		if src.dtype != np.uint8:
			return np.ones(src.shape, dtype=np.uint8)
		srcYCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
		srcHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
		skinPixels = self._R1(src) & self._R2(srcYCrCb) & self._R3(srcHSV)
		return np.asarray(skinPixels, dtype=np.uint8)

class CascadedDetector(Detector):
	"""
	Uses the OpenCV cascades to perform the detection. Returns the Regions of Interest, where
	the detector assumes a face. You probably have to play around with the scaleFactor, 
	minNeighbors and minSize parameters to get good results for your use case. From my 
	personal experience, all I can say is: there's no parameter combination which *just 
	works*.	
	"""
	def __init__(self, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.2, minNeighbors=5, minSize=(30,30)):
		if not os.path.exists(cascade_fn):
			raise IOError("No valid cascade found for path=%s." % cascade_fn)
		self.cascade = cv2.CascadeClassifier(cascade_fn)
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.minSize = minSize
	
	def detect(self, src):
		if np.ndim(src) == 3:
			src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		src = cv2.equalizeHist(src)
		rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
		if len(rects) == 0:
			return np.ndarray((0,))
		rects[:,2:] += rects[:,:2]
		return rects

class SkinFaceDetector(Detector):
	"""
	Uses the SkinDetector to accept only faces over a given skin color tone threshold (ignored for 
	grayscale images). Be careful with skin color tone thresholding, as it won't work in uncontrolled 
	scenarios (without preprocessing)!
	
	"""
	def __init__(self, threshold=0.3, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.2, minNeighbors=5, minSize=(30,30)):
		self.faceDetector = CascadedDetector(cascade_fn=cascade_fn, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
		self.skinDetector = SkinDetector()
		self.threshold = threshold

	def detect(self, src):
		rects = []
		for i,r in enumerate(self.faceDetector.detect(src)):
			x0,y0,x1,y1 = r
			face = src[y0:y1,x0:x1]
			skinPixels = self.skinDetector.detect(face)
			skinPercentage = float(np.sum(skinPixels)) / skinPixels.size
			print skinPercentage
			if skinPercentage > self.threshold:
				rects.append(r)
		return rects
		
if __name__ == "__main__":
	# script parameters
	if len(sys.argv) < 2:
		raise Exception("No image given.")
	inFileName = sys.argv[1]
	outFileName = None
	if len(sys.argv) > 2:
		outFileName = sys.argv[2]
	if outFileName == inFileName:
		outFileName = None
	# detection begins here
	img = np.array(cv2.imread(inFileName), dtype=np.uint8)
	imgOut = img.copy()
	# set up detectors
	#detector = SkinFaceDetector(threshold=0.3, cascade_fn="/home/philipp/projects/opencv2/OpenCV-2.3.1/data/haarcascades/haarcascade_frontalface_alt2.xml")
	detector = CascadedDetector(cascade_fn="/home/philipp/projects/opencv2/OpenCV-2.3.1/data/haarcascades/haarcascade_frontalface_alt2.xml")
	eyesDetector = CascadedDetector(scaleFactor=1.1,minNeighbors=5, minSize=(20,20), cascade_fn="/home/philipp/projects/opencv2/OpenCV-2.3.1/data/haarcascades/haarcascade_eye.xml")
	# detection
	for i,r in enumerate(detector.detect(img)):
		x0,y0,x1,y1 = r
		cv2.rectangle(imgOut, (x0,y0),(x1,y1),(0,255,0),1)
		face = img[y0:y1,x0:x1]
		for j,r2 in enumerate(eyesDetector.detect(face)):
			ex0,ey0,ex1,ey1 = r2
			cv2.rectangle(imgOut, (x0+ex0,y0+ey0),(x0+ex1,y0+ey1),(0,255,0),1)
	# display image or write to file
	if outFileName is None:
		cv2.imshow('faces', imgOut)
		cv2.waitKey(0)
		cv2.imwrite(outFileName, imgOut) 

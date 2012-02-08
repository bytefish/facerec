#    Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
#    Released to public domain under terms of the BSD Simplified license.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are met:
#        * Redistributions of source code must retain the above copyright
#          notice, this list of conditions and the following disclaimer.
#        * Redistributions in binary form must reproduce the above copyright
#          notice, this list of conditions and the following disclaimer in the
#          documentation and/or other materials provided with the distribution.
#        * Neither the name of the organization nor the names of its contributors 
#          may be used to endorse or promote products derived from this software 
#          without specific prior written permission.
#
#    See <http://www.opensource.org/licenses/bsd-license>
import cv2
# cv2 helper
from helper.common import *
from helper.video import *
# add facerec to system path
import sys
sys.path.append("../..")
# facerec imports
from facerec.dataset import DataSet
from facedet.detector import CascadedDetector
from facerec.preprocessing import TanTriggsPreprocessing
from facerec.feature import LBP
from facerec.classifier import NearestNeighbor
from facerec.operators import ChainOperator
from facerec.model import PredictableModel
from facerec.distance import ChiSquareDistance

help_message = '''USAGE: videofacerec.py [<video source>] [<face database>]

Keys:
  ESC   - exit
'''

class App(object):
	def __init__(self, video_src, dataset_fn, face_sz=(130,130), cascade_fn="/home/philipp/projects/opencv2/OpenCV-2.3.1/data/haarcascades/haarcascade_frontalface_alt2.xml"):
		self.face_sz = face_sz
		self.cam = create_capture(video_src)
		ret, self.frame = self.cam.read()
		self.detector = CascadedDetector(cascade_fn=cascade_fn, minNeighbors=5, scaleFactor=1.1)
		# define feature extraction chain & and classifier) 
		feature = ChainOperator(TanTriggsPreprocessing(), LBP())
		classifier = NearestNeighbor(dist_metric=ChiSquareDistance())
		# build the predictable model
		self.predictor = PredictableModel(feature, classifier)
		# read the data & compute the predictor
		self.dataSet = DataSet(filename=dataset_fn,sz=self.face_sz)
		self.predictor.compute(self.dataSet.data,self.dataSet.labels)
			
	def run(self):
		while True:
			ret, frame = self.cam.read()
			# resize the frame to half the original size
			img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
			for i,r in enumerate(self.detector.detect(img)):
				x0,y0,x1,y1 = r
				# get face, convert to grayscale & resize to face_sz
				face = img[y0:y1, x0:x1].copy()
				face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, self.face_sz, interpolation = cv2.INTER_CUBIC)
				# get a prediction
				prediction = self.predictor.predict(face)
				# draw the face area
				cv2.rectangle(img, (x0,y0),(x1,y1),(0,255,0),2)
				# draw the predicted name (folder name...)
				draw_str(img, (x0-20,y0-20), self.dataSet.names[prediction])
				cv2.imshow('img', img)
			# get pressed key
			ch = cv2.waitKey(10)
			if ch == 27:
				break

if __name__ == '__main__':
	import sys
	print help_message
	if len(sys.argv) < 3:
		sys.exit()
	# get params
	video_src = sys.argv[1]
	dataset_fn = sys.argv[2]
	# start facerec app
	App(video_src, dataset_fn).run()

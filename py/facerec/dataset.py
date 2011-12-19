import os as os
import numpy as np
import PIL.Image as Image
import random

class DataSet(object):
	def __init__(self, fileName=None, sz=None):
		self.labels = []
		self.data = []
		self.sz = sz
		if fileName is not None:
			self.load(fileName)

	def shuffle(self):
		idx = np.argsort([random.random() for i in xrange(len(self.labels))])
		self.data = [self.data[i] for i in idx]
		self.labels = self.labels[idx]

	def load(self, path):
		c = 0
		for dirname, dirnames, filenames in os.walk(path):
			for subdirname in dirnames:
				subject_path = os.path.join(dirname, subdirname)
				for filename in os.listdir(subject_path):
					try:
						im = Image.open(os.path.join(subject_path, filename))
						im = im.convert("L")
						# resize to given size (if given)
						if (self.sz is not None) and isinstance(self.sz, tuple) and (len(self.sz) == 2):
							im = im.resize(self.sz, Image.ANTIALIAS)
						self.data.append(np.asarray(im, dtype=np.uint8))
						self.labels.append(c)
					except IOError:
						pass
				c = c+1
		self.labels = np.array(self.labels, dtype=np.int)

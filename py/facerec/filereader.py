import os as os
import numpy as np
import PIL.Image as Image
import util

"""
	Author: philipp <bytefish[at]gmx.de>
	License: BSD (see LICENSE for details)
	Description:
		Classes for Reading from various data sources. Will probably see a Dataset class soon (would make life easier).
"""

class FilesystemReader(object):
	"""
	Arguments:
		data [dim x num_data] image matrix (column-based)
		classes [1 x num_data] classes corresponding to data.
		width [int] width of images
		height [int] height of images
	"""
	def __init__(self, path=None):
		""" Initialize an empty dataset and scan for path if given. """
		self.width, self.height = (0,0)
		self.classes = []
		self.data = []
		self.subjects = []
		try:
			self.load(path)
		except:
			pass
			
	def className(self, num):
		""" Return the name of the class.
		Args:
			num [int] number of the class
		"""
		if num < 0 or num >= len(self.names):
			return 'Class not in Dataset.'
		return self.names[num]

	def shuffle(self):
		self.data, self.classes = util.shuffle(self.data, self.classes)

	def load(self, path):
		""" Load images from a given path (resets stored data).
		
		Args:
			path [string] Path to scan for.
		"""
		self.classes = []
		self.names = []
		images = []
		c = 0 # classes from {0,...,c}		
		for dirname, dirnames, filenames in os.walk(path):
			for subdirname in dirnames:
				subject_path = os.path.join(dirname, subdirname)
				subject_images = []
				subject_classes = []
				for filename in os.listdir(subject_path):
					try:
						im = Image.open(os.path.join(subject_path, filename))
						im = im.convert("L") # convert to greyscale
						subject_images.append(im)
						subject_classes.append(c)
					except IOError:
						pass
				
				if len(subject_images)>0:
					images.extend(subject_images)
					self.classes.extend(subject_classes)
					self.names.append(subdirname)
					c = c+1
		
		if len(images) == 0:
			return (np.empty([0,0], dtype=np.uint8), np.empty([0,0],dtype=np.int))
		
		# take reference width and height from first image
		self.width, self.height = images[0].size
		
		# all images are grayscale, so 8 unsigned bit are sufficient
		self.data = np.empty([self.width*self.height, 0], dtype=np.uint8)
		
		# build image matrix
		for i in range(0, len(images)):
			imarr = np.asmatrix(images[i], dtype=np.uint8)
			try:
				self.data = np.append(self.data, imarr.reshape(-1, 1), axis=1)
			except:
				print "Image %d cannot be added. " % (i)
				del self.classes[i] # remove class for img that can't be added
				# should really make this more intelligent...
				pass

		self.classes = np.array(self.classes, dtype=np.uint8)

	@staticmethod
	def read_image(filename):
		""" Reads an Image from path and returns a grayscale representation. 
		Args:
			filename [string] path to read from
		Returns:
			image data [width*height x 1]
		"""
		try:
			im = Image.open(os.path.join(filename))
			im = im.convert("L") # convert to greyscale
			imarr = np.asmatrix(im, dtype=np.uint8)
			return imarr.reshape(-1,1)
		except IOError:
			pass
		

	def clear(self):
		""" Clear current dataset. """
		self.width, self.height = (0,0)
		self.classes = []
		self.data = []
		self.subjects = []
		

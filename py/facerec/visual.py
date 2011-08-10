import os as os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image

import math as math

""" 
	Author: philipp <bytefish[at]gmx.de>
	License: BSD (see LICENSE for details)
	Description:
		Visualization Module.
	TODO:
			* remove saving within functions (feels very hardcoded)
			* styling plots (set ranges, captions, label, sizes, ...)
			* better combination of plots (model1 vs. model2 vs. model3), my attempt is really weird right now
			* ... (a lot)
"""

class Commons(object):

	@staticmethod
	def normalize(X, low, high):
		""" Normalize a given matrix to range [low,high].
		
		Args:
			X [rows x columns] input data
			low [numeric] lower bound
			high [numeric] upper bound
		"""
		minX = min(X)
		maxX = max(X)
		# Normalize to [0...1].	
		X = X - minX
		X = X / (maxX - minX)
		# Scale to [low...high].
		X = X * (high-low)
		X = X + low
		return X
		
	@staticmethod
	def show(figure):
		""" Shows image from figure or filename.
		
		Args:
			figure: figure or filename
			
		TODO
		"""
		pass


class Layout:
	""" Base Class for Layout
	"""
	font = { 'fontname': 'Tahoma', 'fontsize':10 }
	
class PlotBasic(object):
	
	@staticmethod
	def plot_grayscale(I,  filename, size=None):
		""" Grayscale representation of a given matrix.
		
		Args:
			I [width x height] input image
			filename [string] location to store the plot at
			size [tuple] image dimensions to reshape to (default size(I))
		"""
		if size == None:
			size = I.shape
		width, height = size
		Ig = Commons.normalize(I, 0, 255)
		Ig = Ig.reshape(height, width)
		fig = plt.figure()
		implot = plt.imshow(np.asarray(Ig), cmap=cm.gray)
		fig.savefig(filename, format="png", transparent=False)
	
class XYPlot(BasicPlot):
	pass
	
class PlotSubspace:
	
	@staticmethod
	def plot_weight(model, num_components, dataset, filename, start_component=0, rows = None, cols = None, title="Subplot", color=True):
		"""	Make a subplot of a range of weights.
		
			TODO options to adjust margin, labels, ...
	
		Args:
			W [dim x num_data] Weight matrix from subspace method.
			filename [string] path to store plot 
			start_component [int] component to start from
			num_components [int] number of components to include
			rows [int] rows in this subplot (default sqrt(num_components))
			cols [int] columns in this subplot (default 1)
			size [tuple] (width,height)
			
		Example:
			PlotSubspace.weight(pca.W, (130,100), 16, 
		"""
		W = model.W
		if (rows is None) or (cols is None):
			rows = cols = int(math.ceil(np.sqrt(num_components)))
		num_components = np.min(num_components, W.shape[1])
		print filename
		(width, height) = (dataset.width, dataset.height)
		
		fig = plt.figure()
		for i in range(0, num_components):
			component = Commons.normalize(np.asarray(W[0:,i]), 0, 255)
			component = component.reshape(height,width)
			
			ax0 = fig.add_subplot(rows,cols,i+1)
			# diable 
			plt.setp(ax0.get_xticklabels(), visible=False)
			plt.setp(ax0.get_yticklabels(), visible=False)
			plt.title("%s #%d" % (title, (i+1)), Layout.font)
			if color:
				implot = plt.imshow(np.asarray(component))
			else:
				implot = plt.imshow(np.asarray(component), cmap=cm.grey)
		fig.savefig(filename, format="png", transparent=False)
	


class PlotValidation:
	
	@staticmethod
	def parameter(validators, parameter, filename, xtitle="parameter", xlim=None, title="Accuracy vs. Parameter"):
		""" Creates an Errorbar plot with Parameter (X) vs Validator Accuracy (Y). 
	
		The parameters must be passed as a list (e.g. [10,15,20,25,30]) in case models have more than 1 parameter. 
		It would be ambigous otherwise.
	
		Args:
			validators [dict]
				key [string] title of validators
				value [list] list of validator objects
			parameters [list] Parameter values corresponding to the validators of each model.
		
		Example:
		
			validators = {'Eigenfaces' : [val0, val1, val2], 'Fisherfaces':[val0, val1, val2]}
			parameters = [10, 20, 30]
			
			 parameter_vs_accuracy(validators, parameter, filename="eigenfaces_vs_accuracy.png", xtitle="Eigenfaces", xlim=[0,105], title="Number of Eigenfaces vs. Accuracy")
		"""
		fig = plt.figure()
		legend = []
		for (key,val) in validators.iteritems():
			legend.append(key)
			errs = [v.std_accuracy for v in val]
			y = [v.accuracy for v in val]
			errplot = plt.errorbar(parameter, y, yerr=errs, linestyle='-')
	
		plt.title(title)
		plt.legend(legend, 'upper right', shadow=True, fancybox=True)
		plt.xlabel(xtitle)
		plt.ylabel("Accuracy")
	
		# All plotting functions in matplotlib trigger the autoscale, so
		# call this just before saving. If you want to disable autoscaling:
		# gca().set_autoscale_on(False)
		plt.ylim([0,1])
		if xlim is not None:
			plt.xlim(xlim)
		fig.savefig(filename, format="png", transparent=False)
		
if __name__ == "__main__":
	import filereader
	import models
	data, classes, width, height = filereader.ReaderFilesystem.scan("/home/philipp/facerec/data/c1")
	f = models.Fisherfaces(data, classes)
	PlotSubspace.subplot_ev(f.W, 9, 4, 4, width, height, "test.png")
	Image.open("test.png").show()


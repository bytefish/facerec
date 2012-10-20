from facerec.normalization import minmax

import os as os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image

import math as math


def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }

def plot_gray(X,  sz=None, filename=None):
    if not sz is None:
        X = X.reshape(sz)
    X = minmax(I, 0, 255)
    fig = plt.figure()
    implot = plt.imshow(np.asarray(Ig), cmap=cm.gray)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, format="png", transparent=False)
    
def plot_eigenvectors(eigenvectors, num_components, sz, filename=None, start_component=0, rows = None, cols = None, title="Subplot", color=True):
        if (rows is None) or (cols is None):
            rows = cols = int(math.ceil(np.sqrt(num_components)))
        num_components = np.min(num_components, eigenvectors.shape[1])
        fig = plt.figure()
        for i in range(start_component, num_components):
            vi = eigenvectors[0:,i].copy()
            vi = minmax(np.asarray(vi), 0, 255, dtype=np.uint8)
            vi = vi.reshape(sz)
            
            ax0 = fig.add_subplot(rows,cols,(i-start_component)+1)
            
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax0.get_yticklabels(), visible=False)
            plt.title("%s #%d" % (title, i), create_font('Tahoma',10))
            if color:
                implot = plt.imshow(np.asarray(vi))
            else:
                implot = plt.imshow(np.asarray(vi), cmap=cm.grey)
        if filename is None:
            fig.show()
        else:
            fig.savefig(filename, format="png", transparent=False)
            
def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center') 
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


# using plt plot:
#filename="/home/philipp/facerec/at_database_vs_accuracy_xy.png"
#t = np.arange(2., 10., 1.)
#fig = plt.figure()
#plt.plot(t, r0, 'k--', t, r1, 'k')
#plt.legend(("Eigenfaces", "Fisherfaces"), 'lower right', shadow=True, fancybox=True)
#plt.ylim(0,1)
#plt.ylabel('Recognition Rate')
#plt.xlabel('Database Size (Images per Person)')
#fig.savefig(filename, format="png", transparent=False)
#plt.show()



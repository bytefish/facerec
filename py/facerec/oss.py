import numpy as np

from facerec.distance import AbstractDistance
from facerec.util import asColumnMatrix

class OSS(AbstractDistance):
    """This metric calculates the One-Shot Similarity (OSS) using LDA as the underlying classifier
    
        OSS was originally described in the paper:
 
            Lior Wolf, Tal Hassner and Yaniv Taigman, "The One-Shot Similarity Kernel,"
            IEEE International Conference on Computer Vision (ICCV), Sept. 2009
            http://www.openu.ac.il/home/hassner/projects/Ossk/WolfHassnerTaigman_ICCV09.pdf
        
        This implementation is based on the MATLAB implementation available at:
            
            http://www.openu.ac.il/home/hassner/projects/Ossk/
 
        Copyright 2009, Lior Wolf, Tal Hassner, and Yaniv Taigman
        
        Input:
            XSN [list] A list of samples
    """
    def __init__(self, XSN):
        if XSN is None:
            raise ValueError("XSN cannot be None")
        # Reshape into a column matrix:
        XSN = asColumnMatrix(XSN)
        self.meanXSN = np.mean(XSN, axis=1)
        Sw = np.cov(XSN.T)
        w,v = np.linalg.eigh(Sw)
        idx = np.argsort(-w)
        w = w[idx]
        # Take the largest eigenvalue:
        maxeig = w[0]
        Sw = Sw + 0.1 * np.eye(Sw.shape[0])*maxeig
        self.iSw = np.inv(Sw)
        self.sizeXSN = XSN.shape[1]
        
    def __call__(self, i, j):
        mm = x1 - self.meanXSN
        v = np.dot(self.iSw,mm)
        v = v/np.norm(v)
        v0 = np.dot(v.T,(x1+self.meanXSN))/2.
        score = np.dot(v.T,x2)-v0
        return score
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
from scipy.signal import convolve2d

class LocalDescriptor(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self,X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")
        
    @property
    def neighbors(self):
        return self._neighbors
        
    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % (self._neighbors)

class OriginalLBP(LocalDescriptor):
    def __init__(self):
        LocalDescriptor.__init__(self, neighbors=8)
    
    def __call__(self,X):
        X = np.asarray(X)
        X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
            + (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
            + (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
            + (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
            + (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
            + (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
            + (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
            + (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
        return X
        
    def __repr__(self):
        return "OriginalLBP (neighbors=%s)" % (self._neighbors)

class ExtendedLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius
        
    def __call__(self,X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2*np.pi/self._neighbors
        theta = np.arange(0,2*np.pi,angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny=min(sample_points[:,0])
        maxy=max(sample_points[:,0])
        minx=min(sample_points[:,1])
        maxx=max(sample_points[:,1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
        blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
        # coordinates of origin (0,0) in the block
        origy =  0 - np.floor(min(miny,0))
        origx =  0 - np.floor(min(minx,0))
        # calculate output image size
        dx = xsize - blocksizex + 1
        dy = ysize - blocksizey + 1
        # get center points
        C = np.asarray(X[origy:origy+dy,origx:origx+dx], dtype=np.uint8)
        result = np.zeros((dy,dx), dtype=np.uint32)
        for i,p in enumerate(sample_points):
            # get coordinate in the block
            y,x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y.
            fx = np.floor(x)
            fy = np.floor(y)
            cx = np.ceil(x)
            cy = np.ceil(y)
            # calculate fractional part    
            ty = y - fy
            tx = x - fx
            # calculate interpolation weights
            w1 = (1 - tx) * (1 - ty)
            w2 =      tx  * (1 - ty)
            w3 = (1 - tx) *      ty
            w4 =      tx  *      ty
            # calculate interpolated image
            N = w1*X[fy:fy+dy,fx:fx+dx]
            N += w2*X[fy:fy+dy,cx:cx+dx]
            N += w3*X[cy:cy+dy,fx:fx+dx]
            N += w4*X[cy:cy+dy,cx:cx+dx]
            # update LBP codes        
            D = N >= C
            result += (1<<i)*D
        return result

    @property
    def radius(self):
        return self._radius
    
    def __repr__(self):
        return "ExtendedLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)
        
class VarLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius
        
    def __call__(self,X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2*np.pi/self._neighbors
        theta = np.arange(0,2*np.pi,angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny=min(sample_points[:,0])
        maxy=max(sample_points[:,0])
        minx=min(sample_points[:,1])
        maxx=max(sample_points[:,1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy,0)) - np.floor(min(miny,0)) + 1
        blocksizex = np.ceil(max(maxx,0)) - np.floor(min(minx,0)) + 1
        # coordinates of origin (0,0) in the block
        origy =  0 - np.floor(min(miny,0))
        origx =  0 - np.floor(min(minx,0))
        # Calculate output image size:
        dx = xsize - blocksizex + 1
        dy = ysize - blocksizey + 1
        # Allocate memory for online variance calculation:
        mean = np.zeros((dy,dx), dtype=np.float32)
        delta = np.zeros((dy,dx), dtype=np.float32)
        m2 = np.zeros((dy,dx), dtype=np.float32)
        # Holds the resulting variance matrix:
        result = np.zeros((dy,dx), dtype=np.float32)
        for i,p in enumerate(sample_points):
            # Get coordinate in the block:
            y,x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y:
            fx = np.floor(x)
            fy = np.floor(y)
            cx = np.ceil(x)
            cy = np.ceil(y)
            # Calculate fractional part:
            ty = y - fy
            tx = x - fx
            # Calculate interpolation weights:
            w1 = (1 - tx) * (1 - ty)
            w2 =      tx  * (1 - ty)
            w3 = (1 - tx) *      ty
            w4 =      tx  *      ty
            # Calculate interpolated image:
            N = w1*X[fy:fy+dy,fx:fx+dx]
            N += w2*X[fy:fy+dy,cx:cx+dx]
            N += w3*X[cy:cy+dy,fx:fx+dx]
            N += w4*X[cy:cy+dy,cx:cx+dx]
            # Update the matrices for Online Variance calculation (http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm):
            delta = N - mean
            mean = mean + delta/float(i+1)
            m2 = m2 + delta * (N-mean)
        # Optional estimate for variance is m2/self._neighbors:
        result = m2/(self._neighbors-1)
        return result

    @property
    def radius(self):
        return self._radius
    
    def __repr__(self):
        return "VarLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)

class LPQ(LocalDescriptor):
    """ This implementation of Local Phase Quantization (LPQ) is a 1:1 adaption of the 
        original implementation by Ojansivu V & Heikkilä J, which is available at:
        
            * http://www.cse.oulu.fi/CMV/Downloads/LPQMatlab
            
        So all credit goes to them.
      
      Reference: 
        Ojansivu V & Heikkilä J (2008) Blur insensitive texture classification 
        using local phase quantization. Proc. Image and Signal Processing 
        (ICISP 2008), Cherbourg-Octeville, France, 5099:236-243.

        Copyright 2008 by Heikkilä & Ojansivu
    """
    
    def __init__(self, radius=3):
        LocalDescriptor.__init__(self, neighbors=8)
        self._radius = radius
    
    def euc_dist(self, X):
        Y = X = X.astype(np.float)
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = XX.T
        distances = np.dot(X,Y.T)
        distances *= -2
        distances += XX
        distances += YY
        np.maximum(distances, 0, distances)
        distances.flat[::distances.shape[0] + 1] = 0.0
        return np.sqrt(distances)
        
    def __call__(self,X):
        f = 1.0
        x = np.arange(-self._radius,self._radius+1)
        n = len(x)
        rho = 0.95
        [xp, yp] = np.meshgrid(np.arange(1,(n+1)),np.arange(1,(n+1)))
        pp = np.concatenate((xp,yp)).reshape(2,-1)
        dd = self.euc_dist(pp.T) # squareform(pdist(...)) would do the job, too...
        C = np.power(rho,dd)
        
        w0 = (x*0.0+1.0)
        w1 = np.exp(-2*np.pi*1j*x*f/n)
        w2 = np.conj(w1)
        
        q1 = w0.reshape(-1,1)*w1
        q2 = w1.reshape(-1,1)*w0
        q3 = w1.reshape(-1,1)*w1
        q4 = w1.reshape(-1,1)*w2

        u1 = np.real(q1)
        u2 = np.imag(q1)
        u3 = np.real(q2)
        u4 = np.imag(q2)
        u5 = np.real(q3)
        u6 = np.imag(q3)
        u7 = np.real(q4)
        u8 = np.imag(q4)
        
        M = np.matrix([u1.flatten(1), u2.flatten(1), u3.flatten(1), u4.flatten(1), u5.flatten(1), u6.flatten(1), u7.flatten(1), u8.flatten(1)])
        
        D = np.dot(np.dot(M,C), M.T)
        U,S,V = np.linalg.svd(D)

        Qa = convolve2d(convolve2d(X,w0.reshape(-1,1),mode='same'),w1.reshape(1,-1),mode='same')
        Qb = convolve2d(convolve2d(X,w1.reshape(-1,1),mode='same'),w0.reshape(1,-1),mode='same')
        Qc = convolve2d(convolve2d(X,w1.reshape(-1,1),mode='same'),w1.reshape(1,-1),mode='same')
        Qd = convolve2d(convolve2d(X, w1.reshape(-1,1),mode='same'),w2.reshape(1,-1),mode='same')

        Fa = np.real(Qa)
        Ga = np.imag(Qa)
        Fb = np.real(Qb) 
        Gb = np.imag(Qb)
        Fc = np.real(Qc) 
        Gc = np.imag(Qc)
        Fd = np.real(Qd) 
        Gd = np.imag(Qd)
        
        F = np.array([Fa.flatten(1), Ga.flatten(1), Fb.flatten(1), Gb.flatten(1), Fc.flatten(1), Gc.flatten(1), Fd.flatten(1), Gd.flatten(1)])
        G = np.dot(V.T, F)
        
        t = 0

        # Calculate the LPQ Patterns:
        B = (G[0,:]>=t)*1 + (G[1,:]>=t)*2 + (G[2,:]>=t)*4 + (G[3,:]>=t)*8 + (G[4,:]>=t)*16 + (G[5,:]>=t)*32 + (G[6,:]>=t)*64 + (G[7,:]>=t)*128
        
        return np.reshape(B, np.shape(Fa))
        
    @property
    def radius(self):
        return self._radius
    
    def __repr__(self):
        return "LPQ (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)
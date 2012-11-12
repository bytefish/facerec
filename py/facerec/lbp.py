import numpy as np

class LBPOperator(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self,X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")
        
    @property
    def neighbors(self):
        return self._neighbors
        
    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % (self._neighbors)

class OriginalLBP(LBPOperator):
    def __init__(self):
        LBPOperator.__init__(self, neighbors=8)
    
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

class ExtendedLBP(LBPOperator):
    def __init__(self, radius=1, neighbors=8):
        LBPOperator.__init__(self, neighbors=neighbors)
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
        
class VarLBP(LBPOperator):
    def __init__(self, radius=1, neighbors=8):
        LBPOperator.__init__(self, neighbors=neighbors)
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



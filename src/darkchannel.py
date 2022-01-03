from numpy.core.fromnumeric import argmax
from skimage import io, morphology
import numpy as np
from guided_filter import guided_filter

def darkChannel(m: np.ndarray, r: int = 7):
    res = np.zeros(m.shape[:2])
    res = np.min(m, 2)
    res = morphology.erosion(res, morphology.disk(r))
    return res


def airLight(m: np.ndarray, T: np.ndarray):
    bins = 2000
    ht = np.histogram(T, bins)                              
    d = np.cumsum(ht[0])/float(T.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax]<=0.999:
            break
    loc = argmax(np.mean(m,2)[T>=ht[1][lmax]])
    loc = np.unravel_index(loc,m.shape[:2])
    return m[loc]


def tramsmiss(guided, a=0.95, maxT=0.8):
    """
    t = 1- T/A
    """
    T = np.minimum(guided*a,maxT)
    return T


def deHaze(img, t=0.1):
    dark = darkChannel(img,r=7)
    img_guided = guided_filter(np.min(img,2),dark,81,0.001)
    img_guided =  np.clip(img_guided, 0, 1)
    T = tramsmiss(img_guided)
    A = airLight(img, T)
    J= np.zeros(img.shape)
    for k in range(3):
        J[:,:,k] = (img[:,:,k]-T)/(1-T/A[k]) 
    J = J
    J =  np.clip(J, 0, 1)
    return J

if __name__ == "__main__":
    img = io.imread("./img/tiananmen.jpg")[:,:,:3]/255.0
    dehaze = deHaze(img)
    io.imshow(img)
    io.show()
    io.imshow(dehaze)
    io.show()
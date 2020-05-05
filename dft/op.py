import numpy as np
from global_vars import *

def O(w):
    '''
    Overlap operator
    @input
        w: Fourier weights of shape (S0 * S1 * S2, 1)
    @global variables
        R: lattice vector
    @return
        out: O operator applied to w, where O = (detR)1, with 1 being the identity matrix
    '''
    volume = np.linalg.det(R)
    return volume * w

def L(w):
    '''
    Laplacian operator
    @input
        w: Fourier weights of shape (S0 * S1 * S2, 1)
    @global variables
        R: lattice vector
        G2: lengths squared of G vectors
    @return
        out: L operator applied to w, where L = −(detR)(DiagG2)
    '''
    volume = np.linalg.det(R)
    return -1 * volume * G2 * w

def Linv(w):
    '''
    Inverse of Laplacian operator
    @input
        w: Fourier weights of shape (S0 * S1 * S2, 1)
    @global variables
        R: lattice vector
        G2: lengths squared of G vectors
    @return
        out:  inverse of L operator applied to in, where L = −(detR)(DiagG2) and, by convention, out(1)≡0
    '''
    volume = np.linalg.det(R)
    out = -1 * w / (volume * G2)
    out[0] = 0
    return out

def cI(w):
    '''
    Map from frequency space to real space
    @input
        w: Fourier weights of shape (S0 * S1 * S2, 1)
    @global variables
         S: dimensions of d = 3 dimensional data set
    @return
        out: cI operator applied to in
    '''
    w3 = w.reshape(S[0], S[1], S[2])
    # p3 = np.fft.ifftn(w3, norm='ortho')
    p3 = np.fft.fftn(w3)
    return p3.ravel().reshape(-1, 1)

def cJ(p):
    '''
    Map from real space to frequency space
    @input
        p: real space values of shape (S0 * S1 * S2, 1)
    @global variables
         S: dimensions of d = 3 dimensional data set
    @return
        out:  cJ operator applied to in, where cJ≡cI −1
    '''
    p3 = p.reshape(S[0], S[1], S[2])
    w3 = np.fft.ifftn(p3)
    return w3.ravel().reshape(-1, 1)

if __name__ == '__main__':
    w = np.random.rand(S[0] * S[1] * S[2], 1)
    w_prime = cJ(cI(w))
    print(np.sum(w_prime - w))



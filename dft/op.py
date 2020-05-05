import numpy as np
import global_vars

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
    volume = np.linalg.det(global_vars.R)
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
    volume = np.linalg.det(global_vars.R)
    return -1 * volume * global_vars.G2 * w

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
    volume = np.linalg.det(global_vars.R)
    tmp = np.copy(global_vars.G2)
    tmp[0, 0] = 1  # mute the divide by zero warning
    out = -1 * w / (volume * tmp)
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
    w3 = w.reshape(global_vars.S[0], global_vars.S[1], global_vars.S[2])
    # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
    p3 = np.fft.fftn(w3)  # by numpy's default the fftn is not normalized
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
    p3 = p.reshape(global_vars.S[0], global_vars.S[1], global_vars.S[2])
    # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
    w3 = np.fft.ifftn(p3)  # by numpy's default the ifftn is normalized by 1/N
    return w3.ravel().reshape(-1, 1)

if __name__ == '__main__':
    w = np.random.rand(global_vars.S[0] * global_vars.S[1] * global_vars.S[2], 1)
    w_prime = cJ(cI(w))
    print(np.sum(w_prime - w))



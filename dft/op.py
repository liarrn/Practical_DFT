import numpy as np
import global_vars

def O(w):
    '''
    Overlap operator
    @input
        w: Fourier weights of shape (S0 * S1 * S2, Ns)
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
        w: Fourier weights of shape (S0 * S1 * S2, Ns)
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
        w: Fourier weights of shape (S0 * S1 * S2, Ns)
    @global variables
         S: dimensions of d = 3 dimensional data set
    @return
        out: cI operator applied to in
    '''
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)
    p = np.zeros_like(w, dtype = np.complex128)
    ns = w.shape[1]
    for i in range(ns):
        w3 = w[:, i].reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0])   # (#. z_grid, #. y_grid, #. x_grid)
        # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
        p3 = np.fft.fftn(w3)  # by numpy's default the fftn is not normalized
        p[:, i] = p3.ravel()
    return p

def cIdag(w):
    '''
    cI dagger
    '''
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)
    p = np.zeros_like(w, dtype = np.complex128)
    ns = w.shape[1]
    for i in range(ns):
        w3 = w[:, i].reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0])   # (#. z_grid, #. y_grid, #. x_grid)
        # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
        # cI is not normalized, so here we multiply by the normalization factor added by ifftn to "unnormalize"
        p3 = np.fft.ifftn(w3) * np.prod(global_vars.S)  # by numpy's default the ifftn is normalized by 1/N
        p[:, i] = p3.ravel()
    return p

def cJ(p):
    '''
    Map from real space to frequency space
    @input
        p: real space values of shape (S0 * S1 * S2, Ns)
    @global variables
         S: dimensions of d = 3 dimensional data set
    @return
        out:  cJ operator applied to in, where cJ≡cI −1
    '''
    if len(p.shape) == 1:
        p = p.reshape(-1, 1)
    w = np.zeros_like(p, dtype = np.complex128)
    ns = p.shape[1]
    for i in range(ns):
        p3 = p[:, i].reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0])  # (#. z_grid, #. y_grid, #. x_grid)
        # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
        w3 = np.fft.ifftn(p3)  # by numpy's default the ifftn is normalized by 1/N
        w[:, i] = w3.ravel()
    return w

def cJdag(p):
    '''
    cJ dagger
    '''
    if len(p.shape) == 1:
        p = p.reshape(-1, 1)
    w = np.zeros_like(p, dtype = np.complex128)
    ns = p.shape[1]
    for i in range(ns):
        p3 = p[:, i].reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0])  # (#. z_grid, #. y_grid, #. x_grid)
        # ref: https://docs.scipy.org/doc/numpy/reference/routines.fft.html
        # cJ is normalized by factor 1 / N, fftn by default is not normalized, so here we add the normalization factor by hand
        w3 = np.fft.fftn(p3) / (np.prod(global_vars.S))  # by numpy's default the fftn is not normalized
        w[:, i] = w3.ravel()
    return w

def test_cI():
    w = np.random.rand(np.prod(global_vars.S), 1)
    w_prime = cJ(cI(w))
    assert np.sum(w_prime - w) < 1e-6, 'ERROR in cI and cJ'

def test_cIdag():
    a = np.random.rand(np.prod(global_vars.S), 1)
    b = np.random.rand(np.prod(global_vars.S), 1)
    x = np.dot(a.conj().T, cI(b)).conj()  # (a(dag)Ib)(star)
    y = np.dot(b.conj().T, cIdag(a)) # (b(dag)I(dag)a)
    assert x - y < 1e-5, 'ERROR in cIdag'
    x = np.dot(a.conj().T, cJ(b)).conj()  # (a(dag)Jb)(star)
    y = np.dot(b.conj().T, cJdag(a))  # (b(dag)J(dag)a)
    assert x - y < 1e-5, 'ERROR in cJdag'

if __name__ == '__main__':
    test_cI()
    test_cIdag()



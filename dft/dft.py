import numpy as np
import matplotlib.pyplot as plt
import global_vars
import op
from scipy import linalg
import vis

lattice_center = np.reshape(np.sum(global_vars.R / 2, axis=1), (1, -1))
dr = np.sqrt(np.sum((global_vars.r - lattice_center) ** 2, axis=1))
V = 2 * (dr ** 2)
gb_Vdual = op.cJdag(op.O(op.cJ(V)))


def excVWN(n):
    '''
    VWN parameterization of the exchange correlation energy
    '''
    # Constants
    X1 = 0.75 * (3.0 / (2.0 * np.pi)) ** (2.0 / 3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c

    rs = (4 * np.pi / 3 * n) ** (-1/3)  # Added internal conversion to rs
    x = np.sqrt(rs)
    X = x * x + b * x + c
    out = -1 * X1 / rs + A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b)) - (b * x0) / X0 *
                              (np.log((x - x0)*(x - x0) / X) + 2 * (2 * x0 + b) / Q * np.arctan(Q / (2 * x + b))))
    return out


def excpVWN(n):
    '''
    d/dn deriv of VWN parameterization of the exchange correlation energy
    '''
    # Constants
    X1 = 0.75 * (3.0 / (2.0 * np.pi)) ** (2.0 / 3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c

    rs = (4 * np.pi / 3 * n) ** (-1/3)  # Added internal conversion to rs
    x = np.sqrt(rs)
    X = x * x + b * x + c

    dx = 0.5 / x  # Chain rule needs dx/drho!
    out = dx * (2 * X1 / (rs * x) + A * (2 / x - (2 * x + b) / X - 4 * b / (Q * Q + (2 * x + b) * (2 * x + b))
                                         - (b * x0) / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (2 * x0 + b) / (Q * Q + (2 * x + b) * (2 * x + b)))))
    # Added d(rs)/dn from chain rule from rs to n conv
    out = (-1 * rs / (3 * n)) * out
    return out


def getE(W):
    '''
    get the energy from expansion coefficients for Ns unconstrained wave function
    @input:
        W: Expansion coefficients for Ns unconstrained wave functions, stored as 
            an (S0 * S1 * S2, Ns) matrix
    @global variables
        Vdual: Dual potential coefficients stored as a (S0 * S1 * S2, 1) column vector.
    @return:
        E: Energies summed over Ns states
    '''
    f = 2
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    # each col of W_real represents W's value in sampled real space
    W_real = op.cI(W)
    n = f * op.diagouter(np.dot(W_real, U_inv), W_real)
    E_potential = np.dot(gb_Vdual.T, n)
    E_kinetic = -0.5 * np.trace(np.dot(W.conj().T, op.L(np.dot(W, U_inv)))) * f
    # electrostatic potential (phi) in freqency space
    phi = op.poisson(n, real_phi=False)
    # E_hartree = 0.5 * np.dot(n.conj().T, op.cJdag(op.O(phi)))
    E_hartree = 0.5 * np.dot(op.cJ(n).conj().T, op.O(phi))
    exc = excVWN(n)
    # the Exc is negative here, which seems problematic.
    Exc = np.dot(op.cJ(n).conj().T, op.O(op.cJ(exc)))
    E = (E_potential + E_kinetic + E_hartree + Exc)[0, 0]
    E = E.real
    return E


def H(W):
    '''
    H dot W where H is Harmiton operator
    @input:
        W: Expansion coefficients for Ns unconstrained wave functions, stored as 
            an (S0 * S1 * S2, Ns) matrix
    @global variables
        Vdual: Dual potential coefficients stored as a (S0 * S1 * S2, 1) column vector.
    @return:
        matrix of shape (S0 * S1 * S2, Ns)
    '''
    f = 2
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    # each col of W_real represents W's value in sampled real space
    W_real = op.cI(W)
    n = f * op.diagouter(np.dot(W_real, U_inv), W_real)
    exc = excVWN(n)
    exc_prime = excpVWN(n)
    # electrostatic potential (phi) in freqency space
    phi = op.poisson(n, real_phi=False)
    Veff = gb_Vdual \
        + op.cJdag(op.O(phi)) \
        + op.cJdag(op.O(op.cJ(exc))) \
        + op.diagprod(exc_prime, op.cJdag(op.O(op.cJ(n))))
    HW_potential = op.cIdag(op.diagprod(Veff, op.cI(W)))
    HW_kinetic = -0.5 * op.L(W)
    HW = HW_kinetic + HW_potential
    return HW


def getgrad(W):
    '''
    get gradient dE/dW
    '''
    f = 2
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    HW = H(W)
    t1 = np.dot(op.O(W), U_inv)  # (#. sample points, Ns)
    t2 = np.dot(t1, np.dot(W.conj().T, HW))
    t3 = HW - t2
    t4 = np.dot(t3, U_inv) * f
    return t4


def orthonormalizeW(W):
    '''
    orthonormalize W so that Wdag dot O(W) is identity matrix
    '''
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    U_inv_sqrt = linalg.sqrtm(U_inv)
    W_orthonormal = np.dot(W, U_inv_sqrt)
    return W_orthonormal


def sd(W, niter):
    '''
    perform steepest descents for niter iterations to minimize e
    '''
    alpha = 3e-5
    e = getE(W)
    out_str = 'iteration: {:3d}, energy: {:.6f}'
    print(out_str.format(0, e))
    for i in range(niter):
        dW = getgrad(W)
        W -= alpha * dW
        e = getE(W)
        print(out_str.format(i + 1, e))
    return W

def matdot(a, b):
    '''
    unravel matrix a, b as column vectors, and get their dot product
    It's easy to verify that the dot product equals re(trace(a.conj().T, b))
    '''
    return np.trace(np.dot(a.conj().T, b)).real

def K(w):
    '''
    Preconditioner for conjuncated-gradient optimization
    Here the K = Diag(1/1+G2).
    KW equals apply 1/1+G2 to each row of W
    @input
        w: Fourier weights of shape (S0 * S1 * S2, Ns)
    @global variables
        G2: lengths squared of G vectors
    @return
        K dot W
    '''
    k = 1 / (1 + global_vars.G2)
    return k * w

def pclm(W, niter):
    '''
    perform preconditioned line minimization for niter iterations to minimize e
    '''
    alpha_t = 3e-5
    e = getE(W)
    out_str = 'iteration: {:3d}, energy: {:.6f}, angle cosin: {:.3f}'
    print(out_str.format(0, e, 0))
    d = None  # search direction from last step
    for i in range(niter):
        g = getgrad(W)
        if i > 0:
            # angle test
            cos = matdot(g, d) / (np.linalg.norm(g) * np.linalg.norm(d))
        else:
            cos = 0
        d = -K(g)  # search direction is opposite of gradient
        gt = getgrad(W + alpha_t * d)
        alpha = alpha_t * matdot(g, d) / matdot(g - gt, d)
        W += alpha * d
        e = getE(W)
        print(out_str.format(i + 1, e, cos))
    return W

def pccg(W, niter):
    '''
    perform preconditioned conjugate gradient for niter iterations to minimize e
    1, precondition
    2, line minimizaton/line search
    3, conjugate gradient
    '''
    alpha_t = 3e-5
    e = getE(W)
    out_str = 'iteration: {:3d}, energy: {:.6f}'
    print(out_str.format(0, e, 0))
    d0 = None  # search direction from last step
    g0 = None  # gradient from last step
    for i in range(niter):
        g = getgrad(W)
        if i == 0:
            beta = 0
            d0 = 0
        else:
            beta = matdot(g, K(g)) / matdot(g0, K(g0))
        d = -K(g) + beta * d0  # search direction is opposite of gradient
        g0, d0 = g, d
        gt = getgrad(W + alpha_t * d)
        alpha = alpha_t * matdot(g, d) / matdot(g - gt, d)
        W += alpha * d
        e = getE(W)
        print(out_str.format(i + 1, e))
    return W

def getPsi(W):
    '''
    get the eigenstates psi from non-orthonormal W that minimizes E
    @input W
    @output
        psi: eigenstates
        epsion: eigenvalues
    '''
    W = orthonormalizeW(W)
    Mu = np.dot(W.conj().T, H(W))
    epsilon, D = np.linalg.eig(Mu)
    Psi = np.dot(W, D)
    return epsilon, Psi


def vis_cell_3slice(dat):
    dat = dat.reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0])
    vis.slice(dat, 'xy', global_vars.S[2] // 2)
    vis.slice(dat, 'xz', global_vars.S[1] // 2)
    vis.slice(dat, 'yz', global_vars.S[0] // 2)


def test_all():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * \
        np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    sd(W, 500)
    epsilon, Psi = getPsi(W)
    # in harmonic potential with w = 2, the energy level should be 3, 5, 5, 5
    print('energy levels: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        epsilon[0], epsilon[1], epsilon[2], epsilon[3]))
    W_real = op.cI(W)
    electron_density = W_real.conj() * W_real
    vis_cell_3slice(electron_density[:, 0])  # s orbital
    vis_cell_3slice(electron_density[:, 1])  # p orbital
    vis_cell_3slice(electron_density[:, 2])  # p orbital
    return


def test_getE():
    W = np.random.rand(np.prod(global_vars.S), 4) + 1j * \
        np.random.rand(np.prod(global_vars.S), 4)
    W = orthonormalizeW(W)
    E = getE(W)  # E should be real to machine-precision
    E_prime = np.trace(np.dot(W.conj().T, H(W))).real
    print('E: ', E)
    print('E_prime: ', E_prime)


def test_getgrad():
    ns = 10
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * \
        np.random.rand(np.prod(global_vars.S), ns)
    e0 = getE(W)
    g0 = getgrad(W)

    dW = 1e-6 * (np.random.rand(np.prod(global_vars.S), ns) +
                 1j * np.random.rand(np.prod(global_vars.S), ns))
    de_num = getE(W + dW) - e0
    de_analytical = 2 * np.trace(np.dot(g0.conj().T, dW)).real
    deviation = np.abs(de_num - de_analytical)
    print('numerical result:  {}\nanalytical result: {}\n'.format(
        de_num, de_analytical))
    print('absolute deviation: {}\nrelative deviation: {}\n'.format(
        deviation, deviation / de_num))
    return


def test_orthonormalizeW():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * \
        np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    # Overlap matrix should equal to identity matrix up to machine precison
    Overlap = np.dot(W.conj().T, op.O(W))
    assert np.linalg.norm(np.sum(Overlap - np.identity(ns))
                          ) < 1e-6, 'ERROR in orthonormalizeW'


def test_sd():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * \
        np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    sd(W, 200)

def test_lm():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * \
        np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    sd(W, 20) # use sd in first steps to get to near the minimum
    W = orthonormalizeW(W)
    pccg(W, 200)


def test_sqrtm():
    a = np.random.rand(4, 4)
    a_sqrt = linalg.sqrtm(a)
    a_prime = np.dot(a_sqrt, a_sqrt)
    print(a)
    print(a_prime)
    print(np.sum(a - a_prime))


def test_getPsi():
    W = np.random.rand(np.prod(global_vars.S), 4) + 1j * \
        np.random.rand(np.prod(global_vars.S), 4)
    epsilon, Psi = getPsi(W)
    assert np.trace(np.dot(Psi.conj().T, H(Psi))).real - \
        np.sum(epsilon).real < 1e-6
    # np.dot(Psi.conj().T, op.O(Psi)) should be identity matrix
    return


if __name__ == "__main__":
    # test_getgrad()
    # test()
    # test_orthonormalizeW()
    # test_sd()
    # test_getE()
    # test_getPsi()
    # test_all()
    test_lm()

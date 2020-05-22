import numpy as np
import matplotlib.pyplot as plt
import global_vars
import op
from scipy import linalg
import vis

lattice_center = np.reshape(np.sum(global_vars.R / 2, axis=1), (1, -1))
dr = np.sqrt(np.sum((global_vars.r - lattice_center) ** 2, axis = 1))
V = 2 * (dr ** 2)
gb_Vdual = op.cJdag(op.O(op.cJ(V)))

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
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    W_real = op.cI(W)  # each col of W_real represents W's value in sampled real space
    n = op.diagouter(np.dot(W_real, U_inv), W_real)
    E_potential = np.dot(gb_Vdual.T, n)
    E_kinetic = -0.5 * np.trace(np.dot(W.conj().T, op.L(np.dot(W, U_inv))))
    E = (E_potential + E_kinetic)[0, 0]
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
    HW_kinetic = -0.5 * op.L(W)
    HW_potential = op.cIdag(op.diagprod(gb_Vdual, op.cI(W)))
    HW = HW_kinetic + HW_potential
    return HW

def getgrad(W):
    '''
    get gradient dE/dW
    '''
    U = np.dot(W.conj().T, op.O(W))
    U_inv = np.linalg.inv(U)
    HW = H(W)
    t1 = np.dot(op.O(W), U_inv)  # (#. sample points, Ns)
    t2 = np.dot(t1, np.dot(W.conj().T, HW))
    t3 = HW - t2
    t4 = np.dot(t3, U_inv)
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
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    sd(W, 500)
    epsilon, Psi = getPsi(W)
    # in harmonic potential with w = 2, the energy level should be 3, 5, 5, 5
    print('energy levels: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(epsilon[0], epsilon[1], epsilon[2], epsilon[3]))
    W_real = op.cI(W)
    electron_density = W_real.conj() * W_real
    vis_cell_3slice(electron_density[:, 0])  # s orbital
    vis_cell_3slice(electron_density[:, 1])  # p orbital
    vis_cell_3slice(electron_density[:, 2])  # p orbital
    return

def test_getE():
    W = np.random.rand(np.prod(global_vars.S),4) + 1j * np.random.rand(np.prod(global_vars.S), 4)
    W = orthonormalizeW(W)
    E = getE(W)  # E should be real to machine-precision 
    E_prime = np.trace(np.dot(W.conj().T, H(W))).real
    print('E: ', E)
    print('E_prime: ', E_prime)

def test_getgrad():
    ns = 10
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * np.random.rand(np.prod(global_vars.S), ns)
    e0  = getE(W)
    g0 = getgrad(W)

    dW = 1e-6 * (np.random.rand(np.prod(global_vars.S), ns) + 1j * np.random.rand(np.prod(global_vars.S), ns))
    de_num = getE(W + dW) - e0
    de_analytical = 2 * np.trace(np.dot(g0.conj().T, dW)).real
    deviation = np.abs(de_num - de_analytical)
    print('numerical result:  {}\nanalytical result: {}\n'.format(de_num, de_analytical))
    print('absolute deviation: {}\nrelative deviation: {}\n'.format(deviation, deviation / de_num))
    return 

def test_orthonormalizeW():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    Overlap = np.dot(W.conj().T, op.O(W))  # Overlap matrix should equal to identity matrix up to machine precison
    assert np.linalg.norm(np.sum(Overlap - np.identity(ns))) < 1e-6, 'ERROR in orthonormalizeW'

def test_sd():
    ns = 4
    W = np.random.rand(np.prod(global_vars.S), ns) + 1j * np.random.rand(np.prod(global_vars.S), ns)
    W = orthonormalizeW(W)
    sd(W, 200)

def test_sqrtm():
    a = np.random.rand(4, 4)
    a_sqrt = linalg.sqrtm(a)
    a_prime = np.dot(a_sqrt, a_sqrt)
    print(a)
    print(a_prime)
    print(np.sum(a - a_prime))

def test_getPsi():
    W = np.random.rand(np.prod(global_vars.S),4) + 1j * np.random.rand(np.prod(global_vars.S), 4)
    epsilon, Psi = getPsi(W)
    assert np.trace(np.dot(Psi.conj().T, H(Psi))).real - np.sum(epsilon).real < 1e-6
    # np.dot(Psi.conj().T, op.O(Psi)) should be identity matrix 
    return

if __name__ == "__main__":
    # test_getgrad()
    # test()
    # test_orthonormalizeW()
    # test_sd()
    # test_getE()
    # test_getPsi()
    test_all()

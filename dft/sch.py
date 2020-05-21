import numpy as np
import matplotlib.pyplot as plt
import global_vars
import op

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

def test_getE():
    W = np.random.rand(np.prod(global_vars.S),4) + 1j * np.random.rand(np.prod(global_vars.S), 4)
    E = getE(W)  # E should be real to machine-precision 

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


if __name__ == "__main__":
    test_getgrad()

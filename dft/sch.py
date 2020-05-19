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

if __name__ == "__main__":
    W = np.random.rand(np.prod(global_vars.S),4) + 1j * np.random.rand(np.prod(global_vars.S), 4)
    E = getE(W)

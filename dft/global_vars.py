import numpy as np


def get_meshgrid(nx, ny, nz):
    n = nx * ny * nz
    mesh_real = np.zeros((n, 3), dtype = np.int)
    mesh_recip = np.zeros((n, 3), dtype = np.int)
    for i in range(n):
        mesh_real[i, 0] = i % nx
        mesh_recip[i, 0] = mesh_real[i, 0] - nx if mesh_real[i, 0] > nx / 2 else mesh_real[i, 0]
        mesh_real[i, 1] = (i // nx) % ny
        mesh_recip[i, 1] = mesh_real[i, 1] - ny if mesh_real[i, 1] > ny / 2 else mesh_real[i, 1]
        mesh_real[i, 2] = i // (nx * ny)
        mesh_recip[i, 2] = mesh_real[i, 2] - nz if mesh_real[i, 2] > nz / 2 else mesh_real[i, 2]
    return mesh_real, mesh_recip

S = np.array([10, 10, 10])  # sample grid
M, N = get_meshgrid(*S)  # M: sample grid points in real space; N: sample grid points in reciprocal space
R = np.diag(np.array([6, 6, 6]))  # lattice params. x axis is the 0th col, y axis is the 1st col
r = np.dot( np.dot(M, np.diag( 1 / S )), R.T)  # sample points in lattice
G = 2 * np.pi * np.dot(N, np.linalg.inv(R))  # sample points in reciprocal lattice
G2 = np.sum(G ** 2, axis = 1).reshape(-1, 1)

is_debug = False

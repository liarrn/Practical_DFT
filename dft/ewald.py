import numpy as np
import global_vars
import op
import vis

lattice_center = np.reshape(np.sum(global_vars.R / 2, axis=1), (1, -1))
dr = np.sqrt(np.sum((global_vars.r - lattice_center) ** 2, axis = 1))
gen_gaussian3 = lambda x, sigma: np.exp(-(x ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** (3 / 2)
sum_over_cell = lambda x: np.sum(x) * np.linalg.det(global_vars.R) / \
    (global_vars.S[0] * global_vars.S[1] * global_vars.S[2])  # sum(xdv)
sigma = 0.25
g1 = global_vars.Z * gen_gaussian3(dr, sigma)
n = op.cI(op.cJ(g1) * global_vars.Sf)
n  = np.real(n)

net_charge = sum_over_cell(n)
print('net charge is {:.4f}'.format(net_charge))

vis.slice(n.reshape(global_vars.S[2], global_vars.S[1], global_vars.S[0]), 'xy', global_vars.S[2] // 2)



# phi = op.cI(op.Linv(-4 * np.pi * op.O( op.cJ(n) )))
# # Due to rounding, tiny imaginary parts creep into the solution. Eliminate
# # by taking the real part.
# phi = np.real(phi)

# Unum = (0.5 * np.real(np.dot(op.cJ(phi).conj().T, op.O(op.cJ(n)))))[0, 0]
# Uself = global_vars.Z ** 2 / (2 * np.sqrt(np.pi)) * (1 / sigma) * global_vars.X.shape[0]
# res_deviation = np.abs(Unum - Uself)
# print('Deviation between numerical and analytical results is {:.4f}'.format(res_deviation))

# if global_vars.is_debug:
#     print(Unum)
#     print(Uself)
# # assert res_deviation < 1e-3, 'Deviation between numerical and analytical results is too large'


# print('end')

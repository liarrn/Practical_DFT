import numpy as np
import matplotlib.pyplot as plt
from global_vars import *
import op

lattice_center = np.reshape(np.sum(R / 2, axis=1), (1, -1))
dr = np.sqrt(np.sum((r - lattice_center) ** 2, axis = 1))
gen_gaussian3 = lambda x, sigma: np.exp(-(x ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** (3 / 2)
sum_over_cell = lambda x: np.sum(x) * np.linalg.det(R) / (S[0] * S[1] * S[2])  # sum(xdv)
sigma1, sigma2 = 0.75, 0.5
g1 = gen_gaussian3(dr, sigma1)
g2 = gen_gaussian3(dr, sigma2)
n = g2 - g1

if is_debug:
    print(sum_over_cell(g1))
    print(sum_over_cell(g2))

net_charge = sum_over_cell(n)
print('net charge is {:.4f}'.format(net_charge))
assert net_charge < 1e-3, 'Current charge is not zero'

phi = op.cI(op.Linv(-4 * np.pi * op.O( op.cJ(n) )))
# Due to rounding, tiny imaginary parts creep into the solution. Eliminate
# by taking the real part.
phi = np.real(phi)

Unum = (0.5 * np.real(np.dot(op.cJ(phi).conj().T, op.O(op.cJ(n)))))[0, 0]
Uanal=((1 / sigma1 + 1 / sigma2) / 2 - np.sqrt(2) / np.sqrt(sigma1 ** 2 + sigma2 ** 2)) / np.sqrt(np.pi)
res_deviation = np.abs(Unum - Uanal)
print('Deviation between numerical and analytical results is {:.4f}'.format(res_deviation))
if is_debug:
    print(Unum)
    print(Uanal)

assert res_deviation < 1e-3, 'Deviation between numerical and analytical results is too large'
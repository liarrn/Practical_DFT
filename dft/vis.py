import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def slice(dat, plane, idx):
    '''
    Visualize slice of 3d data by surface plot
    @input:
        dat: 3d np.ndarray
        plane: 'xy', 'yz', 'xz'
        idx: plane index
    '''
    plane2idx = {'yz': 2, 'xz': 1, 'xy': 0}
    assert len(dat.shape) == 3
    assert plane in plane2idx
    assert idx >= 0 and idx < dat.shape[plane2idx[plane]]
    if plane == 'yz':
        dat_slice = dat[:, :, idx]
    elif plane == 'xz':
        dat_slice = dat[:, idx, :]
    else:
        dat_slice = dat[idx, :, :]
    dat_slice = np.squeeze(dat_slice)
    x, y = np.meshgrid(np.arange(dat_slice.shape[1]), np.arange(dat_slice.shape[0]))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(x, y, dat_slice, rstride=1, cstride=1)
    plt.show()
    return
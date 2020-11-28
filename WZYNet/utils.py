# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_plot(X, Y, Z, title=''):
    """
    Plot 3D surface.

    Args:
        X (np.ndarray): Axis X.(After np.meshgrid or should be a $N \\times N$ array.)
        Y (np.ndarray): Axis Y.(After np.meshgrid or should be a $N \\times N$ array.)
        Z (np.ndarray): Axis Z.(Should be a $N \\times N$ array.)
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    fig.suptitle(title)
    plt.show()


import numpy as np


def train(nx, nm):

    """
    y=1: Si la mitad superior del array suma más que la inferior
    Nx = 2^4
    Nm = 2^16
    """

    x = np.zeros((nx, nm))
    y = np.zeros((nm, 1))

    p_up = np.zeros(nm)
    p_down = np.zeros(nm)

    mi = 0

    while mi < nm:
        xi = 0
        while xi < nx:
            r = np.random.rand(1)

            if xi <= (nx / 2):
                p_up[mi] = p_up[mi] + r
            else:
                p_down[mi] = p_down[mi] + r

            x[xi, mi] = r
            xi = xi + 1

        if p_up[mi] > p_down[mi]:
            y[mi] = 1
        else:
            y[mi] = 0

        mi = mi + 1

    return x, y

"""Global-coordinate, cell-averaged seed used in the validated regression."""
import numpy as np

MODES = (3, 5, 7, 11, 13)
COEFFICIENTS = (1., .7, -.5, .3, -.2)


def perturbation_average(x, y, dx, dy, hot_length, ly):
    x = np.asarray(x); y = np.asarray(y)
    k = 2*np.pi/hot_length
    a = np.maximum(x-dx/2, 0.); b = np.minimum(x+dx/2, hot_length)
    envelope = np.where(b > a, ((b-a)/2-(np.sin(k*b)-np.sin(k*a))/(2*k))/dx, 0.)
    gy = sum(w*np.cos(n*np.pi*y/ly)*np.sinc(n*dy/(2*ly))
             for n,w in zip(MODES, COEFFICIENTS))/sum(abs(w) for w in COEFFICIENTS)
    return envelope[:,None]*gy[None,:]

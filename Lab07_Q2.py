__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER. Code adapted from Mark Newman's squarewell.py.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, arange, sqrt, pi, exp
from scipy.integrate import simpson

"""
CONSTANTS
"""
m = 9.1094e-31     # Mass of electron
hbar = 1.0546e-34  # Planck's constant over 2*pi
e = 1.6022e-19     # Electron charge
V0 = 50*e          # initial potential in eV
a = 1e-11          # distance in m
N = 1000           # number of slices
h = 20*a/N         # slice width
cycle = 0          # number to keep track of how many cycles the shooting method takes

"""
FUNCTIONS
"""

def V_h(x):
    """
    Returns potential energy of the harmonic quantum oscillator

    INPUT:
    x [float] is the distance in m

    OUTPUT:
    V [float] is the potential in J
    """
    V = V0 * x**2 / (a**2)
    return V

def V(x):
    """
    Returns potential energy of the anharmonic quantum oscillator

    INPUT:
    x [float] is the distance in m

    OUTPUT:
    V [float] is the potential in J
    """
    V = V0 * x**4 / (a**4)
    return V


def f(r,x,E):
    """
    From Mark Newman's squarewell.py
    """
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2*m/hbar**2)*(V(x)-E)*psi
    return array([fpsi,fphi],float)


def solve(E):
    """
    Adapted from Mark Newman's squarewell.py
    """
    psi = 0.0
    phi = 1.0
    r = array([psi,phi],float)

    for x in arange(-10*a,10*a,h):
        k1 = h*f(r,x,E)
        k2 = h*f(r+0.5*k1,x+0.5*h,E)
        k3 = h*f(r+0.5*k2,x+0.5*h,E)
        k4 = h*f(r+k3,x+h,E)
        r += (k1+2*k2+2*k3+k4)/6
        if cycle==9:
            psis.append(r[0])
    return r[0]

"""
PART A, B, C
"""



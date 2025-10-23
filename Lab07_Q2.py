__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This file provides the python code associated with answering Newman’s Exercise 8.14, where we approximate the 
# energy levels and wavefunctions of a harmonic and anharmonic quantum oscillator. Code adapted from Mark Newman's squarewell.py.

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

def V(x):
    """
    Returns potential energy of the harmonic quantum oscillator

    INPUT:
    x [float] is the distance in m

    OUTPUT:
    V [float] is the potential in J
    """
    V = V0 * x**2 / (a**2)
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
        if cycle==9:  #once we have reached the desired cycle, track psi
            psis.append(r[0])
    return r[0]

"""
QUESTION
"""

###FOR HARMONIC OSCILLATOR
E1 = 0.0 *e
E2 = e
psi2 = solve(E1)

target = e/1000
cycle = 0  #E0 takes 9 cycles to find, so we start with cycle = 0
psis = []
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle +=1

print("E_0 =",E2/e,"eV")
plt.figure()
psis = np.array(psis)
norm = simpson(abs(psis[:500])**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), np.array(psis)/sqrt(2*norm), linewidth=3,alpha=0.3, label='$\psi_{0}$', color ='teal')


E1 = 400.0 *e
E2 = e
psi2 = solve(E1)
cycle=3  #E1 takes 6 cycles to find, so we start with cycle = 3
psis = []
target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle += 1
print("E_1 =",E2/e,"eV")
norm = simpson(np.abs(array(psis[:500]))**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), array(psis) / (-sqrt(2*norm)), linewidth=3,alpha=0.3, label='$\psi_{1}$', color ='coral')


E1 = 600.0 *e
E2 = e
psi2 = solve(E1)
cycle=0  #E2 takes 9 cycles to find, so we start with cycle = 0
psis = []
target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle+=1

print("E_2 =",E2/e,"eV")
norm = simpson(np.abs(array(psis[:500]))**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), psis / sqrt(2*norm), linewidth=3,alpha=0.3, label='$\psi_{2}$', color ='purple')


#plotting analytic wavefunctions
xs = np.linspace(-10*a,10*a,N)
w = sqrt(2*V0 / (m*a**2))
alpha = m*w / hbar
y = sqrt(alpha)*xs

psi_0 = (alpha/pi)**(1/4) * exp(-y**2 / 2)
plt.plot(xs, psi_0, color='teal', ls='--', label='Analytic $\psi_{0}$')

psi_1 = (alpha/pi)**(1/4) * sqrt(2) * y * exp(-y**2 / 2)
plt.plot(xs, psi_1, color='coral', ls='--', label='Analytic $\psi_{1}$')

psi_2 = (alpha/pi)**(1/4) * exp(-alpha*xs**2 / 2) * (2*y**2 - 1) / sqrt(2)
plt.plot(xs, psi_2, color='purple', ls='--', label='Analytic $\psi_{2}$')

plt.ylim(-2e5,2e5)
plt.legend()
plt.grid(alpha=0.5)
plt.xlabel('Distance [m]')
plt.ylabel('Normalised $\Psi$')
plt.title('Normalised Wavefunctions for Harmonic Oscillator')
# plt.savefig('harmonic.pdf')
plt.show()

print('E1 - E0 =', 414.0719165318545*e - 138.02397130603683*e)
print('E2 - E1 =', 690.1198621105397*e - 414.0719165318545*e)
print('The difference between the spacing of these energies is', abs((414.0719165318545*e - 138.02397130603683*e) - (690.1198621105397*e - 414.0719165318545*e)), 'which is smaller than our precision', e/1000)

###FOR ANHARMONIC OSCILLATOR
#redefine V(x) to be anharmonic
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

E1 = 100.0 *e
E2 = e
psi2 = solve(E1)
cycle = 2  #E0 takes 7 cycles to find, so we start with cycle = 2
psis = []

target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle+=1

plt.figure()
print("E_0 =",E2/e,"eV")
norm = simpson(np.abs(array(psis[:500]))**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), psis / sqrt(2*norm), label='$\psi_{0}$', color ='teal')


E1 = 800.0 *e
E2 = 600.0*e
psi2 = solve(E1)
cycle = 4  #E1 takes 5 cycles to find, so we start with cycle = 4
psis = []

target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle+=1

print("E_1 =",E2/e,"eV")
norm = simpson(np.abs(array(psis[:500]))**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), psis / sqrt(2*norm), label='$\psi_{1}$', color ='coral')

E1 = 1500.0 *e
E2 = 1400*e
psi2 = solve(E1)
cycle = 5  #E2 takes 4 cycles to find, so we start with cycle = 5
psis = []

target = e/1000
while abs(E1-E2)>target:
    psi1,psi2 = psi2,solve(E2)
    E1,E2 = E2,E2-psi2*(E2-E1)/(psi2-psi1)
    cycle+=1
    
print("E_2 =",E2/e,"eV")
norm = simpson(np.abs(array(psis[:500]))**2, np.linspace(-10*a,0,500))  #finding normalisation constant with half of the wavefunction
plt.plot(np.linspace(-10*a,10*a,N), psis / sqrt(2*norm), label='$\psi_{2}$', color ='purple')

plt.ylim(-2.5e5,2.5e5)
plt.grid(alpha=0.5)
plt.legend()
plt.xlabel('Distance [m]')
plt.ylabel('Normalised $\Psi$')
plt.title('Normalised Wavefunctions for Anharmonic Oscillator')
# plt.savefig('anharmonic.pdf')
plt.show()

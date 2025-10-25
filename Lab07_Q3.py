__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This code answers Exercise 8.18 from Mark Newman's Computational Physics, where we plot an 
# oscillating chemical reaction.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

"""
FUNCTIONS
"""
def f(r):
    """
    Returns the right hand side of the Brusselator equations
    
    INPUT:
    r [array] is an array of the chemical concentrations
    
    OUTPUT:
    r_new [array] is an array of the Brusselator equation right hand sides
    """
    x = r[0]
    y = r[1]
    
    fx = 1 - (b + 1) * x + a * x ** 2 * y
    fy = b * x - a * x ** 2 * y
    r_new = np.array([fx, fy], float)
    return r_new


def R_nm(r1, r2, n, m):
    """
    Returns R_n,m for the adaptive Bulirsch Stoer method
    
    INPUT:
    R1 [array] is the first Richardson extrapolation
    R2 [array] is the second Richardson extrapolation
    n [integer] is the number of steps
    m [integer] is the Richardson extrapolation indicator
    
    OUTPUT:
    r_nm [array] is the extrapolation estimate
    """
    r_nm = r2[m-2] + (r2[m-2] - r1[m-2]) / ((n / (n-1)) ** (2 * (m-1)) - 1)
    return r_nm


def modified_midpoint(r, n, H):
    """
    Returns r(t+H) using the modified midpoint method
    
    INPUT:
    r [array] contains x, dx/dt, y, dy/dt
    n [integer] is the number of steps
    H [float] is the size of the interval
    
    OUTPUT:
    mm [array] is the modified midpoint solution
    """
    h = H/n
    r = np.copy(r)
    k = r + 0.5 * h * f(r)
    r += f(k) * h
    for i in range(n - 1):
        k += h * f(r)
        r += h * f(k)
    mm = 0.5 * (r + k + 0.5 * h * f(r))
    return mm


def bulirsch(r, t, H):
    """
    Returns the Richardson extrapolation within target accuracy
    
    INPUT:
    r [array] is the initial conditions
    t [float] is the time value
    H [float] is the size of the interval
    
    OUTPUT:
    row [array] is the r array updated at t + H
    """
    
    def row_n(R1, n):
        """
        Returns the n_th row of Richardson extrapolations
        
        INPUT:
        R1 [array] is the first Richardson estimate
        n [integer] is the row number
        
        OUTPUT:
        the Richardson estimate within target accuracy
        """

        if n>8:
            r1 = bulirsch(r, t, H/2)
            return bulirsch(r1, t+H/2, H/2)
        else:
            R2 = [modified_midpoint(r, n, H)]  #get r_n,1
            for m in range(2, n + 1):  #iterate through the rest of row
                R2.append(R_nm(R1, R2, n, m))

            #find error
            R2 = np.array(R2, float)
            error = (R2[n-2] - R1[n-2]) / ((n / (n-1)) ** (2 * (n-1)) - 1)
            error = np.sqrt(error[0] ** 2 + error[1] ** 2)

            target_acc = H * delta
            if error < target_acc:  #check if target accuracy is reached
                ts.append(t + H)
                xs.append(R2[n-1][0])
                ys.append(R2[n-1][1])
                return R2[n-1]
            else:  #re-do if not
                return row_n(R2, n + 1)
    row = row_n(np.array([modified_midpoint(r, 1, H)], float), 2)
    return row

"""
8.18
"""
a = 1
b = 3
t0 = 0
tf = 20
x0 = 0
y0 = 0
delta = 10 ** -10  # target accuracy

r = np.array([x0, y0], float) #initialising arrays
ts = [t0]
xs = [r[0]]
ys = [r[1]]

bulirsch(r, t0, tf - t0)  #compute answer

t, x, y = ts, xs, ys
plt.figure()
plt.plot(t, x, color = 'purple', label = 'Chemical x')
plt.scatter(t, x, color = 'purple', alpha=0.6, label = 'x interval boundaries')
plt.plot(t, y, color = 'coral', label = 'Chemical y')
plt.scatter(t, y, color = 'coral',alpha=0.6, label = 'y inteval boundaries')
plt.xlabel('Time [s]')
plt.ylabel('Chemical Concentration')
plt.title('Brusselator Chemical Reactions')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
# plt.savefig('brussel.pdf', bbox_inches='tight')
plt.show()

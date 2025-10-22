__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def rhs(r: np.ndarray[float]) -> np.ndarray[float]:
    """The right-hand-side of the equations from Exercise 8.8. There is no
    explicit time dependence. This function was taken and slightly modififed
    from Newman_8-8.py from the lab 7 handout.

    Arguments:
    r -- array of [x, vx, y, vy]
    """
    m = 10.0
    l = 2.0

    r2 = r[0] ** 2 + r[2] ** 2
    fx, fy = -m * np.array([r[0], r[2]], float) / (r2 * np.sqrt(r2 + 0.25 * l ** 2))

    return np.array([r[1], fx, r[3], fy], float)

"""
PART A)
"""
a = 0.0
b = 10.0
N = 1000
h = (b - a) / N

tpoints = np.arange(a, b, h)
xpoints = []
vxpoints = [] 
ypoints = []
vypoints = [] 

r = np.array([1.0, 0.0, 0.0, 1.0], float)
for t in tpoints:
    xpoints.append(r[0])
    vxpoints.append(r[1])
    ypoints.append(r[2])
    vypoints.append(r[3])
    k1 = h * rhs(r)
    k2 = h * rhs(r + 0.5 * k1)
    k3 = h * rhs(r + 0.5 * k2)
    k4 = h * rhs(r + k3)
    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

"Plotting Trajectory of a Ball Bearing Around a Space Rod"
plt.figure()

# Plotting trajectory of a ball bearing around a space rod
plt.plot(xpoints, ypoints, ls = ':', color = 'Teal')

# Labels
plt.title("Trajectory of a Ball Bearing Around a Space Rod", fontsize = 12)
plt.xlabel("X Position $(m)$", fontsize = 12)
plt.ylabel("Y Position $(m)$", fontsize = 12)

plt.grid()

plt.savefig('Figures\\Trajectory_of_a_Ball_Bearing_Around_a_Space_Rod.pdf')
plt.show()

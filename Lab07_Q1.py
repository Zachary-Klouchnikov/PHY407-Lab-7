__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This code uses both an adaptive and non-adaptive Runge-Kutta 4th order method
# to solve the equations of motion for a ball bearing orbiting a space rod.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

from time import time
from collections.abc import Callable

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def rk4(f: Callable[[np.ndarray], np.ndarray], r: np.ndarray, t: np.ndarray,
        h: float) -> tuple:
    """Runge-Kutta 4th Order Method for solving ODEs.
    
    Arguments:
    f -- function representing the ODE (dr/dt = f(r, t))
    r -- array for r values
    t -- array for t values
    h -- step size
    """
    x_points = np.array([], float)
    y_points = np.array([], float)

    xdot_points = np.array([], float)
    ydot_points = np.array([], float)

    for t in t:
        x_points = np.append(x_points, r[0])
        y_points = np.append(y_points, r[1])
        xdot_points = np.append(xdot_points, r[2])
        ydot_points = np.append(ydot_points, r[3])

        k_1 = h * f(r)
        k_2 = h * f(r + k_1 / 2)
        k_3 = h * f(r + k_2 / 2)
        k_4 = h * f(r + k_3)
        r += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return x_points, y_points, xdot_points, ydot_points

def rk4_step(f: Callable[[np.ndarray], np.ndarray], r: np.ndarray,
             h: float) -> np.ndarray:
    """Single step of the Runge-Kutta 4th Order Method.
    
    Arguments:
    f -- function representing the ODE (dr/dt = f(r, t))
    r -- array for r values
    h -- step size
    """
    k1 = h * f(r)
    k2 = h * f(r + k1 / 2)
    k3 = h * f(r + k2 / 2)
    k4 = h * f(r + k3)

    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk4_adaptive(f: Callable[[np.ndarray], np.ndarray], r: np.ndarray,
                 t: np.ndarray, h: float, target_error: float) -> tuple:
    """Adaptive Runge-Kutta 4th Order Method for solving ODEs.
    
    Arguments:
    f -- function representing the ODE (dr/dt = f(r, t))
    r -- array for r values
    t -- array for t values
    h -- initial step size
    target_error -- desired error tolerance
    """
    x_points = np.array([], float)
    y_points = np.array([], float)

    xdot_points = np.array([], float)
    ydot_points = np.array([], float)

    t_points = np.array([], float)

    while t[0] < t[-1]:
        # Compute one full and two half-steps
        r_full = rk4_step(f, r, h)
        r_half = rk4_step(f, r, h / 2)
        r_half = rk4_step(f, r_half, h / 2)

        # Error estimation
        error_x = (r_full[0] - r_half[0]) / 30
        error_y = (r_full[1] - r_half[1]) / 30

        # Adjust step size
        rho = h * target_error / np.sqrt(error_x ** 2 + error_y ** 2)

        if rho >= 1: # Accept the step
            t[0] += h
            r = r_half
            x_points = np.append(x_points, r[0])
            y_points = np.append(y_points, r[1])
            xdot_points = np.append(xdot_points, r[2])
            ydot_points = np.append(ydot_points, r[3])
            t_points = np.append(t_points, t[0])
            h *= rho ** 0.25

        else: # Reject the step
            h *= rho ** 0.25 

    return x_points, y_points, xdot_points, ydot_points, t_points

def rhs(r: np.ndarray[float]) -> np.ndarray[float]:
    """The right-hand-side of the equations from Exercise 8.8. There is no
    explicit time dependence. This function was taken and slightly modififed
    from Newman_8-8.py from the lab 7 handout.

    Arguments:
    r -- array of [x, y, vx, vy]
    """
    m = 10.0
    l = 2.0

    r2 = r[0] ** 2 + r[1] ** 2
    fx, fy = -m * np.array([r[0], r[1]], float) / (r2 * np.sqrt(
        r2 + 0.25 * l ** 2))

    return np.array([r[2], r[3], fx, fy], float)

"""
PART A) AND B)
"""
"Non-Adaptive Scheme"
h = 0.001 # Step size
t = np.arange(0, 10, h, float) # Time array
r = np.array([1.0, 0.0, 0.0, 1.0], float) # Initial conditions

# Solve ODE using non-adaptive RK4
start = time()
r_not_adaptive = rk4(rhs, r, t, h) 
end = time()
print(f"Non-Adaptive Scheme took {end - start} seconds")

"Adaptive Scheme"
h = 0.01 # Initial step size
t = np.arange(0, 10, h, float) # Time array
target_error = 1e-6
r = np.array([1.0, 0.0, 0.0, 1.0], float) # Initial conditions

# Solve ODE using adaptive RK4
start = time()
r_adaptive = rk4_adaptive(rhs, r, t, h, target_error) 
end = time()
print(f"Adaptive Scheme took {end - start} seconds")

"Plotting Trajectory of a Ball Bearing Around a Space Rod"
plt.figure()

# Plotting trajectory of a ball bearing around a space rod
plt.plot(r_not_adaptive[0], r_not_adaptive[1], ls = ':', color = 'Teal', label = "Non-Adaptive RK4")
plt.plot(r_adaptive[0], r_adaptive[1], ls = ':', color = 'Coral', label = "Adaptive RK4")

# Labels
plt.title("Trajectory of a Ball Bearing Around a Space Rod", fontsize = 14)
plt.xlabel("X Position $(m)$", fontsize = 14)
plt.ylabel("Y Position $(m)$", fontsize = 14)

plt.legend(fontsize = 14)
plt.grid()

# plt.savefig('Figures\\Trajectory of a Ball Bearing Around a Space Rod.pdf')
plt.show()

"""
PART C)
"""
"Plotting Time Step Sizes of the Adaptive RK4 Method"
plt.figure()

# Plotting time step sizes of the adaptive RK4 method
plt.plot(r_adaptive[4][1:-1], r_adaptive[4][2:] - r_adaptive[4][1:-1], ls = '-', color = 'Teal')

# Labels
plt.title("Time Step Sizes of the Adaptive RK4 Method", fontsize = 12)
plt.xlabel("Time $(s)$", fontsize = 12)
plt.ylabel("Time Step Size $(s)$", fontsize = 12)

plt.grid()

# Limits
plt.xlim(0, 10)

# plt.savefig('Figures\\Time Step Sizes of the Adaptive RK4 Method.pdf')
plt.show()

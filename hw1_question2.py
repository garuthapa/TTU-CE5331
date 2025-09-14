# Qn.2 using TDMA to solve equation of 1D convection-diffusion problem
import numpy as np
import matplotlib.pyplot as plt
import time

# Defined Constants as provided in problem
U = 0.75  # convective velocity 
G = 0.50  # diffusion coefficient 
Q = 0.0   # source

L = 1.0   # size of domain
N_values = [12, 24, 48, 72, 96]

# Boundary conditions
f0 = 1.0  # f(0) = 1 when x=0
fL = 0.0  # f(L) = 0 when x=L

def analytical_sol(x, U, G, L, f0, fL):
    return f0 + (fL - f0) * ((np.exp((U/G)*x) - 1) / (np.exp((U/G)*L) - 1))

# TDMA solving function
def tdma_solver(a_sub, a_0, a_add, b):
    n = len(b)
    a_str = np.zeros(n)
    b_str = np.zeros(n)
    f= np.zeros(n)

    # Step 1: Forward Elimination 
    a_str[0] = a_0[0]
    b_str[0] = b[0]
    for i in range(1, n):
        div = a_sub[i] / a_str[i-1]
        a_str[i] = a_0[i] - div * a_add[i-1]
        b_str[i] = b[i] - div * b_str[i-1]
    # Step 2: Backward Substitution 
    f[n-1] = b_str[n-1] / a_str[n-1]
    for i in range(n - 2, -1, -1):
        f[i] = (b_str[i] - a_add[i] * f[i+1]) / a_str[i]
    return f

# Collectors for each plots (errors and h_values for error plot, elpsed_times for tiem vs N plot)
errors = []
h_values = []
elapsed_times = [] 

# Plot 1: Numerical Solutions (comparision of plot between f(x) vs. N) along with analytical or exact solution
plt.figure(figsize=(8, 5))
x_fine = np.linspace(0, L, 1000)
f_exact_fine = analytical_sol(x_fine, U, G, L, f0, fL)
plt.plot(x_fine, f_exact_fine, 'k-', label='Analytical Solution (Exact solution)')

for N in N_values:
    h = L / N  # Grid spacing
    h_values.append(h)
    x_nodes = np.linspace(0, L, N+1)
    interior_nodes = N - 1
    
    # Interior coefficients using TDMA
    a_sub = np.full(interior_nodes, (G / h**2) + (U / (2*h)))
    a_0 = np.full(interior_nodes, -(2 * G / h**2))
    a_add = np.full(interior_nodes, (G / h**2) - (U / (2*h)))

    # RHS terms for b is 0 for all nodes i= 1 to N-1 as Q=0 in question but since, f(0)=1 and f(L)=0 are boundary terms:
    # a_minus[0] terms with f(0) goes to RHS at node i=1 and a_plus[-1] terms with f(L) goes to RHS at node i= N-1.
    b = np.zeros(N-1)
    b = np.zeros(N-1)
    b[0] -= a_sub[0] * f0
    b[-1] -= a_add[-1] * fL

    # Recording time for TDMA solver to plot time vs N graph
    start_time = time.perf_counter()
    f_h = tdma_solver(a_sub, a_0, a_add, b)
    elapsed_time = time.perf_counter() - start_time
    elapsed_times.append(elapsed_time)

    # Assembling full solution
    f_numerical = np.concatenate(([f0], f_h, [fL]))

    # Plot the numerical solution for the N values
    plt.plot(x_nodes, f_numerical, 'o--', label=f'N = {N}')

    # Error Study
    f_exact = analytical_sol(x_nodes, U, G, L, f0, fL)
    error_vector = f_numerical - f_exact
    error = np.linalg.norm(error_vector, 2)/ np.sqrt(len(f_numerical)) #Root mean square error is used here, by dividing sqrt(N+1) instead of multiplying sqrt(N+1) that might be error in question.
    errors.append(error)

# Plot 1: Plot between f(x) vs x for different N values
plt.title('Numerical Solutions for 1D Convection-Diffusion [Plot between f(x) vs. x with different N values] including Analytical solution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Error Plot (RMS Error vs h) in log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors, 'o-')
plt.title('Log-log Error Plot: $||f_h - f_{exact}||_2/ sqrt {N+1}$ vs. h')
plt.xlabel('h (Grid Spacing)')
plt.ylabel('$||f_h - f_{exact}||_2/ sqrt {N+1}$')
plt.grid(True, which="both", ls="--")
slope = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
plt.text(h_values[0], errors[0], f'Slope ≈ {slope:.2f}', fontsize=12, verticalalignment='bottom')
plt.show()

# Plot 3: Calculation Time vs. N
plt.figure(figsize=(8, 6))
plt.plot(N_values, elapsed_times, 'o-')
plt.title('Calculation Time vs. N (TDMA Complexity)')
plt.xlabel('N')
plt.ylabel('Elapsed Time (s)')
plt.grid(True)
plt.show()

# Added print statement to display time for N=24 to compare with non-linear diffusion in question 3
print(f"Calculation time for N=24: {elapsed_times[1]:.6f} seconds")
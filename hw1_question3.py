# Qn.3 solving non-linear diffusion problem and comparing with Qn. 2
import numpy as np
import matplotlib.pyplot as plt
import time

# Defined Parameters for non-linear diffusion problem
U = 0.0
L = 1.0
N = 24 
h= L/ N

# Boundary conditions at x=0 and x=L
f0 = 1.0  # f(0) = 1
fL = 0.0  # f(L) = 0
x_nodes = np.linspace(0, L, N+1)

# Analytical solutions
def analytical_Qequal_0(x):
    return -1 + np.sqrt(4 - 3*x)

def analytical_Qequal_0point1(x):
    return -1 + np.sqrt(4 - 2*x - x**2)

def analytical_Qequal_0point1x(x):
    return -1 + np.sqrt(4 - (8/3)*x - (x**3)/3)

# TDMA solver function using code from Qn.2
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

#defining non-linear diffusion G(f)= 0.1+0.1f
G = lambda f: 0.1 + 0.1 * f

#Q values as provided in the problem
Qequal0 = lambda x: np.zeros_like(x)
Q0point1 = lambda x:0.1*np.ones_like(x)
Q0point1x= lambda x: 0.1 * x

# Fixed-point iteration function for the nonlinear problem
def solve_nonlinear(Q_func, G, N):
    h = L / N
    # Initial guess for f (taking linear interpolation between boundaries f0 and fL)
    f_current = np.linspace(f0, fL, N + 1)
    
    max_iter = 200
    tol = 1e-3
    
    for i in range(max_iter):
        f_old = f_current.copy()
        # Recalculating G based on the previous iteration's solution
        G_at_nodes = G(f_current)
        interior_node = N - 1

        # Initialize the arrays for the half-node values of G
        G_avg_sub = np.zeros(interior_node)
        G_avg_add = np.zeros(interior_node)

        for j in range(interior_node):
            # Calculate G at the i-1/2 interface 
            G_avg_sub[j] = (G_at_nodes[j] + G_at_nodes[j+1]) / 2.0
    
            # Calculate G at the i+1/2 interface 
            G_avg_add[j] = (G_at_nodes[j+1] + G_at_nodes[j+2]) / 2.0
            
        a_sub = G_avg_sub/ (h**2)
        a_add = G_avg_add/ (h**2)
        a_0 = -(a_sub + a_add)
        
        # The right-hand side is b=-Q(x)
        b = -Q_func(x_nodes[1:-1])
        
        # Using boundary conditions: f(0)=1 and f(L)=0
        b[0] -= a_sub[0] * f0
        b[-1] -= a_add[-1] * fL
        
        # Solving linear system for the new solution
        f_x = tdma_solver(a_sub, a_0, a_add, b)

        # Updating solution vector with boundary conditions
        f_current = np.concatenate(([f0], f_x, [fL]))
        
        # Checking for convergence
        change = np.linalg.norm(f_current - f_old, 2)
        if change < tol:
            print(f"Converged in {i+1} iterations for Q = {Q_func.__name__}")
            break
    
    return f_current, f_old, i + 1

# Solving for all three cases of Q(x)
N_for_plot = 24
x_nodes = np.linspace(0, L, N_for_plot + 1)

Q_cases = [
    ("Q = 0", Qequal0, analytical_Qequal_0), 
    ("Q = 0.1", Q0point1, analytical_Qequal_0point1),
    ("Q = 0.1x", Q0point1x, analytical_Qequal_0point1x)
]  

labels = ['Q = 0', 'Q = 0.1', 'Q = 0.1x']
colors = ['r', 'b', 'g']
linestyles = ['-', '-', '-']

plt.figure(figsize=(8, 5))

# Solving and plotting for each 3 Q case alogside analytical solution
for i, (label, Q_func, exact_func) in enumerate(Q_cases):
    f_h, _, num_iterations = solve_nonlinear(Q_func, G, N_for_plot)

    # Plot 1: numerical solution (non-linear diffusion)
    plt.plot(x_nodes, f_h, color=colors[i], linestyle=linestyles[i], marker='o', label=f'Numerical: {label}')
    
    # Plot 2: analytical solution
    f_exact = exact_func(x_nodes)
    plt.plot(x_nodes, f_exact, color=colors[i], linestyle='-', linewidth=1, label=f'Analytical: {label}')

plt.title('Solutions of numerical result of 1D Nonlinear Diffusion with G(f)= 0.1+0.1f and N=24 vs. Analytical solution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Calculation time for N=24 for nonlinear diffusion with Q=0 to compare with linear diffusion in Qn.2
start_time = time.perf_counter()
solve_nonlinear(Qequal0, G, N)
elapsed_time = time.perf_counter() - start_time

print(f"Calculation time for N={N} (Q=0): {elapsed_time:.6f} seconds")

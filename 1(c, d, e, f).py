import numpy as np
import matplotlib.pyplot as plt
import time
import os

#Source term S(x, y) given in the problem
def source_term(x, y):
    return 6 * (1 - x**2) * (2*y - 1) - 4*y**3 + 6*y**2 - 2

#Analytical solution T(x,y) given in the problem
def analytical_solution(x, y):
    return (1 - x**2) * (2*y**3 - 3*y**2 + 1)

# Directory to save result of all the plots
output_dir = "output_plots_Problem_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Numerical Solver Functions-Jacobi, Gauss-Seidel, SOR
#1.JAcobi iterative method
def jacobi(T_init, S, h, max_iterations=25000, tolerance=1e-4):
    T = T_init.copy() #Starting with initial grid
    T_new = T.copy()
    residuals = []
    N = T.shape[0] #Sets grid resolution so that it can be used for any grid size

    for k in range(max_iterations):
        T_old = T.copy()
        for i in range(1, N - 1):
            for j in range(N):
                if j > 0 and j < N - 1: # Interior points
                    T_new[i,j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - h**2 * S[i,j])
                elif j == 0: # Bottom Neumann boundary
                    T_new[i,j] = 0.25 * (T[i+1,j] + T[i-1,j] + 2*T[i,j+1] - h**2 * S[i,j])
                elif j == N-1: # Top Neumann boundary
                    T_new[i,j] = 0.25 * (T[i+1,j] + T[i-1,j] + 2*T[i,j-1] - h**2 * S[i,j])
        
        T = T_new.copy()
       #Convergence check using infinity norm to find largest change in T
        residual = np.linalg.norm(T - T_old, ord=np.inf)
        residuals.append(residual) #collects residuals for each iteration for change in T or error
        if residual < tolerance:
            return T, k + 1, residuals 
    print(f"Jacobi did not converge within {max_iterations} iterations for N={T.shape[0]}.")
    return T, k + 1, residuals

#2.Gauss seidel iterative method
def gauss_seidel(T_init, S, h, max_iterations=25000, tolerance=1e-4):
    T = T_init.copy()
    residuals = []
    N = T.shape[0]
    #distinction between using (k+1) (new) and (k) (old) values is handled implicitly updating a single array in place within the same iteration
    for k in range(max_iterations):
        T_old = T.copy()
        for i in range(1, N - 1): #Outer loop sweeps from top to bottom row
            for j in range(N): #Inner loop sweeps from left to right in a row
                if j > 0 and j < N - 1: # Interior points
                    T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - h**2 * S[i, j])
                elif j == 0: # Bottom Neumann boundary
                    T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + 2*T[i, j+1] - h**2 * S[i, j])
                elif j == N - 1: # Top Neumann boundary
                    T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + 2*T[i, j-1] - h**2 * S[i, j])
        #Convergence check using infinity norm to find largest change in T
        residual = np.linalg.norm(T - T_old, ord=np.inf)
        residuals.append(residual)
        if residual < tolerance:
            return T, k + 1, residuals      
    print(f"Gauss-Seidel did not converge within {max_iterations} iterations for N={T.shape[0]}.")
    return T, k + 1, residuals

#3.SOR iterative method
def sor(T_init, S, h, alpha, max_iterations=25000, tolerance=1e-4):
    T = T_init.copy()
    residuals = []
    N = T.shape[0]

    for k in range(max_iterations):
        T_old = T.copy()
        for i in range(1, N - 1): #Outer loop sweeps from top to bottom row
            for j in range(N): #Inner loop sweeps from left to right in a row
                t_ij_old = T[i, j]
                if j > 0 and j < N - 1: # Interior points
                    t_sor = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - h**2 * S[i,j])
                elif j == 0: # Bottom Neumann boundary
                    t_sor = 0.25 * (T[i+1,j] + T[i-1,j] + 2*T[i,j+1] - h**2 * S[i,j])
                elif j == N - 1: # Top Neumann boundary
                    t_sor = 0.25 * (T[i+1,j] + T[i-1,j] + 2*T[i,j-1] - h**2 * S[i,j])
                T[i, j] = (1 - alpha) * t_ij_old + alpha * t_sor

        residual = np.linalg.norm(T - T_old, ord=np.inf)
        residuals.append(residual)
        if residual < tolerance:
            return T, k + 1, residuals     
    print(f"SOR did not converge within {max_iterations} iterations for N={T.shape[0]}.")
    return T, k + 1, residuals

# Function to calculate optimal alpha (relaxation factor) for SOR based on grid size N
# this is derived for model problem solving 2D poisson equation on unit square with uniform grid that minimizes largest eigenvalue (spectral radius) that controls the speed of convergence.
def calculate_optimal_alpha(N):
    alpha = 2 / (1 + np.sin(np.pi / (N - 1)))
    print(f"Calculated optimal alpha for N={N} is {alpha:.4f}")
    return alpha
#if alpha is greater than above derived value (optimal)then method will be unstable and fail to converge.

# Main Execution Function
def main():
    # Problem 1(c): Contours and Centerline Plots (using Gauss-Seidel iterative solver)
    print("\nRunning Problem 1(c) using Gauss-Seidel Solver")
    Ns = [10, 20, 40]
    fig_contours, axes_contours = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig_centerline, ax_centerline = plt.subplots(figsize=(8, 6))
    print("\nCalculating Global RMS Error for different grid sizes:")
    for i, N in enumerate(Ns):
       #Creating grid
        h = 1.0 / (N - 1) #Grid spacing
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        #Initializing temperature field and applying boundary conditions
        T0 = np.zeros((N, N))
        T0[0, :] = analytical_solution(X[0,0], y)
        T0[-1, :] = analytical_solution(X[-1,0], y)
        S = source_term(X, Y)
        # Solve using Gauss-Seidel and analytical solution(getting values from function defined above)
        T_num, _, _ = gauss_seidel(T0.copy(), S, h)
        T_ana = analytical_solution(X,Y)

        # Calculate and print global RMS error for problem 1 (c)
        err_c = np.sqrt(np.mean((T_num - T_ana)**2))
        print(f"  N={N}, RMS Error={err_c:.4e}")

        # Plotting temperature contour
        cs = axes_contours[i].contourf(X.T, Y.T, T_num.T, levels=20, cmap='viridis')
        fig_contours.colorbar(cs, ax=axes_contours[i])
        axes_contours[i].set_title(f"Numerical N={N}")
        axes_contours[i].set_xlabel("x")
        if i == 0: axes_contours[i].set_ylabel("y")
        # Plotting temperature along centerline x=0.5
        center_idx = N // 2
        ax_centerline.plot(y, T_num[center_idx, :], 'o--', label=f"N={N} (Numerical)")
    #Adding analytical solution to plots for comparison
    T_analytical_c = analytical_solution(X, Y)
    cs_ana = axes_contours[3].contourf(X.T, Y.T, T_analytical_c.T, levels=20, cmap='viridis')
    fig_contours.colorbar(cs_ana, ax=axes_contours[3])
    axes_contours[3].set_title("Analytical")
    axes_contours[3].set_xlabel("x")
    fig_contours.suptitle("Temperature Contours (Gauss-Seidel)", fontsize=16)

    ax_centerline.plot(y, analytical_solution(0.5, y), 'k-', label="Analytical")
    ax_centerline.set_title("Problem 1(c): Centerline Temperature at x=0.5 (Gauss-Seidel)")
    ax_centerline.set_xlabel("y")
    ax_centerline.set_ylabel("T(0.5, y)")
    ax_centerline.legend()
    ax_centerline.grid(True)

    # Problem 1(d): Spatial Convergence Rate (using SOR with optimal alpha)
    print("\nRunning Problem 1(d) using SOR Solver for Accuracy")
    Ns_d = [10, 20, 40, 80, 160]
    errors = []
    hs = []
    for N in Ns_d:
        #Creating grid
        h = 1.0 / (N - 1)
        hs.append(h)
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        #Initializing temperature and source term
        T0 = np.zeros((N, N))
        T0[0, :] = analytical_solution(X[0,0], y)
        T0[-1, :] = analytical_solution(X[-1,0], y)
        S = source_term(X, Y)
        
        # Getting optimal alpha for specific grid size from above function and formula
        optimal_alpha = calculate_optimal_alpha(N)
        # Using the SOR solver with the optimal alpha (relaxation value)
        T_num, _, _ = sor(T0, S, h, optimal_alpha)
        T_ana = analytical_solution(X, Y)
        
        # Using Relative L2 Norm for a more robust error metric
        err = np.linalg.norm(T_num - T_ana) / np.linalg.norm(T_ana)
        errors.append(err)
        print(f"N={N}, h={h:.4f}, Relative L2 Error={err:.4e}")
    #plotting convergence result
    fig_d = plt.figure(figsize=(8, 6))
    plt.loglog(hs, errors, 'o-', label='Relative L2 Error')
    p = np.polyfit(np.log(hs), np.log(errors), 1)
    slope = p[0] #slope of 2 means error decreases by a factor of 4 when h is halved
    plt.loglog(hs, np.exp(p[1]) * np.array(hs)**p[0], '--', label=f'Fit Slope = {slope:.2f}')
    plt.title("Problem 1(d): Spatial Convergence Rate (SOR with Optimal $\\alpha$)")
    plt.xlabel("Grid Spacing (h)")
    plt.ylabel("Global Error")
    plt.grid(True, which="both")
    plt.legend()

    # Problem 1(e) & (f): Residuals and Timing Comparison (N=40) 
    print("\nRunning Problem 1(e) and 1(f)")
    N = 40
    h = 1.0 / (N - 1)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    T0 = np.zeros((N, N))
    T0[0, :] = analytical_solution(X[0,0], y)
    T0[-1, :] = analytical_solution(X[-1,0], y)
    S = source_term(X, Y)

    optimal_alpha_40 = calculate_optimal_alpha(N)
    #calculating time taken by each iterative method to converge
    t_start = time.time()
    _, jac_iters, jac_res = jacobi(T0.copy(), S, h)
    t_jac = time.time() - t_start

    t_start = time.time()
    _, gs_iters, gs_res = gauss_seidel(T0.copy(), S, h)
    t_gs = time.time() - t_start
    
    t_start = time.time()
    _, sor_iters, sor_res = sor(T0.copy(), S, h, optimal_alpha_40)
    t_sor = time.time() - t_start
    #1(f): comapring computational time for three solvers and printing values in table format
    print("\nProblem 1(f): Computational Time Comparison (N=40)")
    print(f"{'Method':<15} | {'Iterations':<12} | {'Time (s)':<10}")
    print("-" * 45)
    print(f"{'Jacobi':<15} | {jac_iters:<12} | {t_jac:<10.4f}")
    print(f"{'Gauss-Seidel':<15} | {gs_iters:<12} | {t_gs:<10.4f}")
    print(f"{'SOR (optimal alphas)':<15} | {sor_iters:<12} | {t_sor:<10.4f}")
    #1(e): residual vs iteration plot for three iterative solvers
    fig_e = plt.figure(figsize=(8, 6))
    plt.semilogy(jac_res, label=f"Jacobi ({jac_iters} iters)")
    plt.semilogy(gs_res, label=f"Gauss-Seidel ({gs_iters} iters)")
    plt.semilogy(sor_res, label=f"SOR ($\\alpha$={optimal_alpha_40:.2f}, {sor_iters} iters)")
    plt.title("Problem 1(e): Residual vs. Iteration (N=40)")
    plt.xlabel("Iteration")
    plt.ylabel("Infinity Norm of Residual")
    plt.grid(True, which="both")
    plt.legend()
    #Sacing all plots in output directory created above
    fig_contours.savefig(os.path.join(output_dir, "problem_1_c_contours.png"))
    fig_centerline.savefig(os.path.join(output_dir, "problem_1_c_centerline.png"))
    fig_d.savefig(os.path.join(output_dir, "problem_1_d_convergence (1e-4).png"))
    fig_e.savefig(os.path.join(output_dir, "problem_1_e_residuals.png"))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
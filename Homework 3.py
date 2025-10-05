import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Creating output directory for plots
OUTPUT_DIR = 'output_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tridiagonal matrix (TDMA) solving function
def tdma_solver(a_sub, a_0, a_add, b):
    n = len(b)
    a_str = np.zeros(n) # Stores modified main diagonal
    b_str = np.zeros(n) # Stores modified RHS
    f = np.zeros(n)     # Solution vector

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

# 1D unsteady transport equation solver class 
class TransportSimulation:
    def __init__(self, L=40.0, x_start=5.0, x_end=45.0, N=500, dt=0.025, U=1.0, k=0.01, t_initial=10.0):
        #Initializing the simulation parameters
        self.L = L
        self.x_start = x_start
        self.x_end = x_end
        self.N = N  # Number of spatial points
        self.dt = dt
        self.U = U
        self.k = k
        self.t_initial = t_initial

        self.dx = (self.x_end - self.x_start) / (N - 1)
        self.x = np.linspace(self.x_start, self.x_end, N)
        self.phi = self.initial_condition(self.x, self.t_initial)
        
        # Boundary conditions
        self.phi[0] = 0
        self.phi[-1] = 0

    def initial_condition(self, x, t):
        #Calculating the initial condition at t=10.
       return (1/np.sqrt(0.4*np.pi)) * np.exp(-2.5*(x-10)**2)
    
    #calculating the analytical solution for both cases
    def analytical_solution(self, x, t):
        if self.k > 0:  # Case A: Advection-Diffusion
            denominator = np.sqrt(4 * np.pi * self.k * t)
            numerator = np.exp(-(x - self.U * t)**2 / (4 * self.k * t))
            return numerator / denominator
        else:  # Case B: Pure Advection
            return (1/np.sqrt(0.4*np.pi)) * np.exp(-2.5*(x - self.U*t)**2)
   
    #1. Forward euler integration scheme
    def solve(self, t_end, method='forward_euler'):
        #Solving the transport equation until t_end using the specified method.
        t = self.t_initial
        history = {t: self.phi.copy()}
        solver_method = getattr(self, method)
        
        num_steps = int(round((t_end - t) / self.dt))
        for _ in range(num_steps):
            self.phi = solver_method()
        # Storing the final state
        t = self.t_initial + (num_steps * self.dt)
        history[t] = self.phi.copy()
        
        return history

    def forward_euler(self):
        phi_at_n = self.phi.copy()
        phi_at_n_plus_1 = np.zeros_like(phi_at_n)
        
        C = self.U * self.dt / (2*self.dx)  # Courant number 
        D = self.k * self.dt / self.dx**2 # Diffusion number

        # Loop over the interior points to calculate the solution at time n+1
        for i in range(1, self.N - 1):
            phi_at_n_plus_1[i] = (1- 2*D)* phi_at_n[i] + (C + D) * (phi_at_n[i-1])+ (-C + D)*(phi_at_n[i+1])
        
        # Boundary conditions are fixed at 0 (value  at the boundaries must be zero for all time t, so after we calculate new state of the system at time n+1, I am enforcing this condition on that new state)
        phi_at_n_plus_1[0] = 0
        phi_at_n_plus_1[-1] = 0
        return phi_at_n_plus_1
    
    #2. Backward euler integration scheme
    def backward_euler(self):
        n_interior = self.N - 2 # Size of the interior system
        C = self.U * self.dt / (2 * self.dx)
        D = self.k * self.dt / self.dx**2

        # Setting up the LHS matrix diagonals for A
        # Coeff of phi[i-1] at n+1: (-D - C)
        a_sub = np.full(n_interior, -D - C)
        # Coeff of phi[i] at n+1: (2*D)
        a_0   = np.full(n_interior, 1 + 2 * D)
        # Coeff of phi[i+1] at n+1: (-D + C)
        a_add = np.full(n_interior, -D + C)
        
        # Setting up the RHS vector, which is the solution at the current time step, phi_at_n
        rhs_d = self.phi[1:-1] 

        # Solving the system for the interior points at time n+1
        phi_interior_at_n_plus_1 = tdma_solver(a_sub, a_0, a_add, rhs_d)
        
        # Assembling the full solution vector for time n+1
        phi_at_n_plus_1 = np.zeros(self.N)
        phi_at_n_plus_1[1:-1] = phi_interior_at_n_plus_1

        # Boundary conditions are fixed at 0
        phi_at_n_plus_1[0] = 0
        phi_at_n_plus_1[-1] = 0
        return phi_at_n_plus_1
    
    #3. Crack-Nicholson integration scheme
    def crank_nicholson(self):
        n_interior = self.N - 2 # Size of the interior system
        C = self.U * self.dt / (4 * self.dx)
        D = self.k * self.dt / (2 * self.dx**2)

        # Setting up the LHS matrix diagonals 
        # Coeff of phi[i-1] at n+1: (-D - C)
        a_sub = np.full(n_interior, -D - C)
        # Coeff of phi[i] at n+1: (1 + 2*D)
        a_0   = np.full(n_interior, 1 + 2 * D)
        # Coeff of phi[i+1] at n+1: (-D + C)
        a_add = np.full(n_interior, -D + C)

        # Calculating the RHS vector based on the solution at time n
        phi_at_n = self.phi
        rhs_d = np.zeros(n_interior)
        
        for i in range(1, self.N - 1):
            # Index for rhs_d is i-1 since it's for the interior system for time step n
            rhs_d[i-1] = (D + C) * phi_at_n[i-1] + (1 - 2*D) * phi_at_n[i] + (D - C) * phi_at_n[i+1]

        # Solving the system for the interior points at time n+1
        phi_interior_at_n_plus_1 = tdma_solver(a_sub, a_0, a_add, rhs_d)

        # Assembling the full solution vector for time n+1
        phi_at_n_plus_1 = np.zeros(self.N)
        phi_at_n_plus_1[1:-1] = phi_interior_at_n_plus_1
        
        # Boundary conditions are fixed at 0
        phi_at_n_plus_1[0] = 0
        phi_at_n_plus_1[-1] = 0
        return phi_at_n_plus_1

# Analytical and numerical solution plotting functions for each tasks
def run_task2():
    #Solving and plotting for Task 2: phi vs. x at different time snapshots for all three cases and comparison with analytical solution
    print("\nRunning Task 2 (Part 1): Plotting phi vs. x")
    times_to_plot = [20, 30, 40]
    methods = ['forward_euler', 'backward_euler', 'crank_nicholson']
    cases = {'A': {'k': 0.01}, 'B': {'k': 0.0}}
    
    # Using parameters that ensure stability for all cases (especially forward euler)
    N = 801
    dt = 0.005

    for case_name, params in cases.items():
        for t_snap in times_to_plot:
            solutions = {}
            plt.figure(figsize=(10, 6))         
             # Analytical solution
            sim_analytical = TransportSimulation(N=N, dt=dt, k=params['k'])
            analytical_sol = sim_analytical.analytical_solution(sim_analytical.x, t_snap)
            plt.plot(sim_analytical.x, analytical_sol, 'k-', label='Analytical', linewidth=2)

            for method in methods:
                sim = TransportSimulation(N=N, dt=dt, k=params['k'])
                history = sim.solve(t_snap, method=method)
                
                actual_time = min(history.keys(), key=lambda t: abs(t - t_snap))
                solutions[method] = history[actual_time]
                
                plt.plot(sim.x, solutions[method], '--', label=f'{method.replace("_", " ").title()}')

            plt.title(f'Case {case_name} (k={params["k"]}): $\phi$ vs. x at t={t_snap}')
            plt.xlabel('x')
            plt.ylabel('$\phi$')
            plt.grid(True)
            plt.legend()
            
            filename = f'task2_case_{case_name}_t_{t_snap}.png'
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.close() 

    print(f"Task 2 plots saved in '{OUTPUT_DIR}/' directory.")

#for fixed case: crack-nicholson integration scheme, three timesnapshots in one plot with analytical solution
def run_task2_combined_snapshots():
    print("\nRunning Task 2 (Part 2): Plotting Combined Time Snapshots")
    times_to_plot = [20, 30, 40]
    method = 'crank_nicholson'
    cases = {'A': {'k': 0.01}, 'B': {'k': 0.0}}
    colors = ['blue', 'green', 'red']

    N = 801
    dt = 0.005

    for case_name, params in cases.items():
        plt.figure(figsize=(12, 7))
        sim_base = TransportSimulation(N=N, dt=dt, k=params['k'])

        for i, t_snap in enumerate(times_to_plot):
            # Analytical Solution for all three time snapshot
            analytical_sol = sim_base.analytical_solution(sim_base.x, t_snap)
            plt.plot(sim_base.x, analytical_sol, color=colors[i], linestyle='-', label=f'Analytical t={t_snap}')
            # Numerical Solution for all three time snapshot
            sim = TransportSimulation(N=N, dt=dt, k=params['k'])
            history = sim.solve(t_snap, method=method)
            actual_time = min(history.keys(), key=lambda t: abs(t - t_snap))
            plt.plot(sim.x, history[actual_time], color=colors[i], linestyle='--', label=f'Crank-Nicolson t={t_snap}')

        plt.title(f'Combined Snapshots for Case {case_name} (Crank-Nicolson vs. Analytical)')
        plt.xlabel('x')
        plt.ylabel('$\phi$')
        plt.grid(True)
        plt.legend()
        
        filename = f'task2_combined_timesnapshots_nicholsoncase_{case_name}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
    print(f"Task 2 combined snapshot plots saved in '{OUTPUT_DIR}/' directory.")

#Solving and plotting for Task 3: phi vs. t at x= 15 and 25 for all three integartion schemes with analytical references different time snapshots for all three cases and comparison with analytical solution
def run_task3():
    print("\nRunning Task 3: Plotting phi vs. t")
    N = 500
    dt = 0.01
    t_end = 40.0
    x_points = [15, 25]
    methods = ['forward_euler', 'backward_euler', 'crank_nicholson']
    
    for x_val in x_points:
        plt.figure(figsize=(10, 6))
        
        sim_base = TransportSimulation(N=N, dt=dt, k=0.01)
        x_idx = np.abs(sim_base.x - x_val).argmin()

        # Analytical solution over time
        times_analytical = np.linspace(sim_base.t_initial, t_end, 400)
        analytical_phi_t = sim_base.analytical_solution(sim_base.x[x_idx], times_analytical)
        plt.plot(times_analytical, analytical_phi_t, 'k-', label='Analytical', linewidth=2)
        
        for method in methods:
            sim = TransportSimulation(N=N, dt=dt, k=0.01)
            
            # Run simulation step-by-step to record history
            t = sim.t_initial
            phi_history_at_x = [sim.phi[x_idx]]
            time_history = [t]
            
            num_steps = int(round((t_end - t) / dt))
            for _ in range(num_steps):
                sim.phi = getattr(sim, method)()
                t += dt
                phi_history_at_x.append(sim.phi[x_idx])
                time_history.append(t)
            
            plt.plot(time_history, phi_history_at_x, '--', label=f'{method.replace("_", " ").title()}')

        plt.title(f'Case A: $\phi$ vs. t at x = {x_val}')
        plt.xlabel('t')
        plt.ylabel('$\phi$')
        plt.grid(True)
        plt.legend()
        
        filename = f'task3_x_{x_val}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
    print(f"Task 3 plots saved in '{OUTPUT_DIR}/' directory.")

# Plotting for Task 4: Spatial convergence analysis with best case so far thats Crack-Nicholson case.
def run_task4():
    print("\nRunning Task 4: Spatial Convergence Analysis")
    
    resolutions = [51, 101, 201, 401, 801]
    dx_values = []
    errors = []
    
    dt = 1e-4  # small dt to minimize temporal error
    t_eval = 30.0
    method = 'crank_nicholson' #most accurate method so far

    for N in resolutions:
        dx = (45.0 - 5.0) / (N - 1)
        dx_values.append(dx)
       
        #Running the simulation
        sim = TransportSimulation(N=N, dt=dt, k=0.01)
        history = sim.solve(t_eval, method=method)
        #Getting the numerical solution at t=20
        actual_time = min(history.keys(), key=lambda t: abs(t - t_eval))
        numerical_sol = history[actual_time]
        
        # Evaluating the analytical solution on the same grid points
        analytical_sol = sim.analytical_solution(sim.x, actual_time)

        error = np.sqrt(np.sum((numerical_sol - analytical_sol)**2) / N)
        errors.append(error)
        print(f"N={N:<4}, dx={dx:<.4f}, Error={error:.2e}")
        
    log_dx = np.log(dx_values)
    log_error = np.log(errors)
    slope, intercept = np.polyfit(log_dx, log_error, 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(dx_values, errors, 'o-', label=f'Numerical Results (Observed Slope = {slope:.2f})')
    
    theoretical_error = np.exp(intercept) * (np.array(dx_values) / dx_values[0])**2
    plt.loglog(dx_values, theoretical_error, 'r--', label='Theoretical 2nd Order Convergence')
    
    plt.title(f'Spatial Convergence for {method.replace("_", " ").title()} (Case A)')
    plt.xlabel('Grid Spacing ($\Delta x$)')
    plt.ylabel('L2 Norm of Error')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis() 
    plt.legend()
    
    filename = 'task4_spatial_convergence.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"\n Observed spatial convergence rate: {slope:.2f}")
    print(f"Plot 1 (fit all points) for Task 4 plot saved in '{OUTPUT_DIR}/' directory.")

    # PLOT 2: Slope calculated using only last 3 points to get slope closer to 2
    dx_last3 = dx_values[-3:]
    errors_last3 = errors[-3:]
    slope_last3, _ = np.polyfit(np.log(dx_last3), np.log(errors_last3), 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(dx_values, errors, 'o-', label=f'Numerical Results (Slope from last 3 points = {slope_last3:.2f})')
    
    # Draw reference line anchored to the first point of the fit (the 3rd data point overall)
    theoretical_error_last3 = errors[-3] * (np.array(dx_last3) / dx_last3[0])**2
    plt.loglog(dx_last3, theoretical_error_last3, 'r--', label='Theoretical 2nd Order Convergence')

    plt.title('Task 4: Spatial Convergence (Fit Using Last 3 Points)')
    plt.xlabel('Grid Spacing ($\Delta x$)')
    plt.ylabel('L2 Norm of Error')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task4_spatial_convergence_last_3_points.png'))
    plt.close()
    print(f"Plot 2 (fit last 3 points) saved. Observed slope: {slope_last3:.2f}")

# Plotting for Task 5: Temporal convergence analysis for all three methods
def run_task5():
    print("\nRunning Task 5: Temporal Convergence Analysis")
    N = 2001 
    dts = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    methods = ['forward_euler', 'backward_euler', 'crank_nicholson']
    t_eval = 30.0
    
    print(f"Time Steps (dt): {dts}")
    plt.figure(figsize=(12, 8))
    convergence_results = {}

    for method in methods:
        dt_values = []
        errors = []
        # Using a high-resolution analytical solution as a "perfect" reference
        sim_ref = TransportSimulation(N=N, dt=min(dts), k=0.01) # Base sim for grid
        analytical_sol_ref = sim_ref.analytical_solution(sim_ref.x, t_eval)

        for dt in dts:
            dt_values.append(dt)
            sim = TransportSimulation(N=N, dt=dt, k=0.01)
            history = sim.solve(t_eval, method=method)
            actual_time = min(history.keys(), key=lambda t: abs(t - t_eval))
            numerical_sol = history[actual_time]
            error = np.sqrt(np.sum((numerical_sol - analytical_sol_ref)**2) / N)
            errors.append(error)

        # Plotting and Analysis
        log_dt = np.log(dt_values)
        log_error = np.log(errors)
        slope, intercept = np.polyfit(log_dt, log_error, 1)
        convergence_results[method] = slope
        p = plt.loglog(dt_values, errors, 'o-', label=f'{method.replace("_", " ").title()} (Observed Slope = {slope:.2f})')
        #theoretical lines are drawn from last point of each numerical result for hree integration schemes to check the slope separately for each cases.
        color = p[0].get_color()
        theoretical_slope = 2 if method == 'crank_nicholson' else 1
        
        # Using the last data point to anchor the theoretical line for better visualization
        theoretical_error = errors[-1] * (np.array(dt_values) / dt_values[-1])**theoretical_slope
        plt.loglog(dt_values, theoretical_error, linestyle='--', color=color, label=f'Theoretical {theoretical_slope}st/nd Order Rate')

    plt.title('Temporal Convergence Analysis (Case A)')
    plt.xlabel('Time Step Size ($\Delta t$)')
    plt.ylabel('L2 Norm of Error')
    plt.grid(True, which="both", ls="--")
    plt.gca().invert_xaxis()
    plt.legend()
    
    filename = 'task5_temporal_convergence.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"\nTask 5 plot saved in '{OUTPUT_DIR}/' directory.")

# Discussion for Task 6: Stability analysis
def run_task6():
    print("\nTask 6: Stability Discussion")
    N=401
    dx = (45 - 5) / (N - 1)
    k=0.01
    # The most restrictive stability condition for Forward Euler in this case
    dt_diffusion_limit = dx**2 / (2 * k)
    
    # saving a summary to your results file
    with open(os.path.join(OUTPUT_DIR, 'results.txt'), "a") as f:
        f.write("\nTask 6: Stability Summary\n")
        f.write(f"Case A: FE is conditionally stable for dt <= {dt_diffusion_limit:.4f} for N={N}). BE and CN are unconditionally stable.\n")
        f.write(f"Case B: FTCS scheme for FE is unconditionally unstable. BE and CN are unconditionally stable.\n")

# Task 7: Computational cost comparison
def run_task7():
    print("\nRunning Task 7: Computational Cost Comparison")
    N = 1001
    dt = 0.001
    t_end = 40.0
    methods = ['forward_euler', 'backward_euler', 'crank_nicholson']
    method_names = [m.replace('_', ' ').title() for m in methods]

    print(f"Timing comparison for N={N}, dt={dt}, t_end={t_end}\n")
    
    results = {}
    for method in methods:
        sim = TransportSimulation(N=N, dt=dt, k=0.01)
        start_time = time.time()
        sim.solve(t_end, method=method)
        end_time = time.time()
        
        total_time = end_time - start_time
        num_steps = (t_end - sim.t_initial) / dt
        time_per_step = total_time / num_steps
        
        results[method] = {'total_time': total_time, 'time_per_step': time_per_step}
    
    # Plotting Total Runtime
    total_times = [results[m]['total_time'] for m in methods]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(method_names, total_times, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel('Time (seconds)')
    plt.title('Total Runtime Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center')
    filename = 'task7_total_runtime.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

    # Plotting Cost Per Step
    per_step_costs_ms = [results[m]['time_per_step'] * 1000 for m in methods] # in ms
    plt.figure(figsize=(10, 6))
    bars = plt.bar(method_names, per_step_costs_ms, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel('Time per Step (milliseconds)')
    plt.title('Computational Cost Per Time Step')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f} ms', va='bottom', ha='center')
    filename = 'task7_cost_per_step.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Task 7 plots saved in '{OUTPUT_DIR}/' directory.")

# Main execution block
if __name__ == '__main__':
    run_task2()
    run_task2_combined_snapshots()
    run_task3()
    run_task4()
    run_task5()
    run_task6()
    run_task7()

   
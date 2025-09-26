import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri #for plotting triangular grids
import os
from scipy.interpolate import griddata #for interpolating data onto different set  of points
from tqdm import tqdm #for creating progress bars for long iterative loop

# Pre-calculating the geometric properties for all the triangles in different meshes
def preprocess_mesh(coords, elements):
    num_elements = len(elements)
    areas = np.zeros(num_elements) #array to store area of each triangle
    circumcenters = np.zeros((num_elements, 2)) #array to store circumcenter (x,y)coordinate of each triangle
    neighbors = -np.ones((num_elements, 3), dtype=int) #array to store IDs of 3 neighbours for each triangle. -1 indicates no neighbor (boundary edge) on that edge or side of triangle.
    edge_lengths = np.zeros((num_elements, 3)) #array to store lengths of each 3 edges of the triangle
    edge_to_elements = {}  #creates dictionary to map each edge to the elements that share it (uses tuple of 2 node IDs that form edge of triangle and values will be list of triangle IDs sharing that edge)

    for i, elem in enumerate(elements):
        v1, v2, v3 = coords[elem[0]], coords[elem[1]], coords[elem[2]]
        # Area of the triangle using shoelace formula to get area of polygon from vertices or coordinates provided.
        areas[i] = 0.5 * np.abs(v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1]) + v3[0]*(v1[1]-v2[1]))
        # Circumcenter calculation from geometric proeprties of triangle vertices
        D = 2 * (v1[0]*(v2[1] - v3[1]) + v2[0]*(v3[1] - v1[1]) + v3[0]*(v1[1] - v2[1]))
        if abs(D) < 1e-12: D = 1e-12
        c_x = ((v1[0]**2 + v1[1]**2)*(v2[1] - v3[1]) + (v2[0]**2 + v2[1]**2)*(v3[1] - v1[1]) + (v3[0]**2 + v3[1]**2)*(v1[1] - v2[1])) / D
        c_y = ((v1[0]**2 + v1[1]**2)*(v3[0] - v2[0]) + (v2[0]**2 + v2[1]**2)*(v1[0] - v3[0]) + (v3[0]**2 + v3[1]**2)*(v2[0] - v1[0])) / D
        circumcenters[i] = [c_x, c_y]
        
        # Edge lengths and finding neighbors
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])] #edges are defined from coordinate file and element file node IDs forming a triangle
        for j, edge in enumerate(edges):
            p_start, p_end = coords[edge[0]], coords[edge[1]]
            edge_lengths[i, j] = np.linalg.norm(p_end - p_start) #numpy function to calculate distance directly using pythagorean formula
            sorted_edge = tuple(sorted(edge)) #dictionary for edge
            if sorted_edge in edge_to_elements:
                edge_to_elements[sorted_edge].append(i) #if edge already in dictionary, its shared edge with another triangle
            else:
                edge_to_elements[sorted_edge] = [i] #if edge is not shared, new entry for edge in the dictionary.

    for i, elem in enumerate(elements):
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for j, edge in enumerate(edges):
            sorted_edge = tuple(sorted(edge))
            elements_on_edge = edge_to_elements[sorted_edge] #finding edge thats shared by 2 triangles
            if len(elements_on_edge) == 2:
                #if shared edge, one is current triangle 'i' and other is neighbour triangle, just finding neighbour ids
                neighbor_idx = elements_on_edge[0] if elements_on_edge[1] == i else elements_on_edge[1]
                neighbors[i, j] = neighbor_idx #storing found neighbour ids in this array
    
    mesh_data = {
        "num_elements": num_elements, "areas": areas, "circumcenters": circumcenters,
        "neighbors": neighbors, "edge_lengths": edge_lengths, "coords": coords, "elements": elements
    }
    return mesh_data
# Using SOR Solver as this iterative is used for all problems
def solve_fvm_sor(mesh, alpha, tol, max_iter=20000):
    T_num = np.zeros(mesh["num_elements"])
    iterations = max_iter
    residuals = []
    for k in tqdm(range(max_iter), desc=f"SOR (ω={alpha:.2f}, tol={tol:.1e})"): #best alpha is used below as required for best convergence as for unstructured mesh to find best alpha formula is abit impossible
        T_old = T_num.copy()
        for i in range(mesh["num_elements"]):
            t_i_k = T_num[i]
            num_sum, den_sum = 0, 0
            for j in range(3):
                neighbor = mesh["neighbors"][i, j]
                edge_len = mesh["edge_lengths"][i, j]
                if neighbor != -1: #Interior edge
                    dist = np.linalg.norm(mesh["circumcenters"][i] - mesh["circumcenters"][neighbor])
                    a_ij = edge_len / dist if dist > 1e-12 else 0 #just for stability check
                    num_sum += a_ij * T_num[neighbor]
                else: #Boundary edge
                    v1_idx, v2_idx = mesh["elements"][i, j], mesh["elements"][i, (j + 1) % 3]
                    midpoint = (mesh["coords"][v1_idx] + mesh["coords"][v2_idx]) / 2
                    dist = np.linalg.norm(mesh["circumcenters"][i] - midpoint)
                    a_ij = edge_len / dist if dist > 1e-12 else 0 #just for stability check
                den_sum += a_ij
            source_term = mesh["areas"][i]
            t_i_gs = (source_term + num_sum) / den_sum if den_sum > 1e-12 else 0
            T_num[i] = (1 - alpha) * t_i_k + alpha * t_i_gs
        
        residual = np.linalg.norm(T_num - T_old, ord=np.inf)
        residuals.append(residual)
        if residual < tol:
            iterations = k + 1
            break
    return T_num, iterations, residuals

# Main Script for all problems
# Analytical solution as defined in the problem
def analytical_solution_p2(x, y, num_terms=40):
    T = np.zeros_like(np.broadcast_to(x, y.shape) if np.ndim(x) < np.ndim(y) else np.broadcast_to(y, x.shape), dtype=float)
    m_vals = np.arange(1, 2 * num_terms, 2) #{1,3,5....num_terms as defined}
    n_vals = np.arange(1, 2 * num_terms, 2) #{1,3,5....num_terms as defined}
    for m in m_vals:
        for n in n_vals:
            term = (np.sin(np.pi * n * x) * np.sin(np.pi * m * y)) / (m * n * (m**2 + n**2))
            T += term
    return T * (16 / np.pi**4)

# Setup Directories and output path for saving plots
script_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(script_dir, "output_plots_Problem_2")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Problem 2(c): Compare how fast the different iterative solvers work by plotting iterations taken to converge vs residual
print("\nRunning Problem 2(c): Solver Comparison (N~40 per side)")
N_c = 40
coords_c = np.loadtxt(os.path.join(script_dir, f'coordinates_{N_c}.input'), delimiter=',')
elements_c = np.loadtxt(os.path.join(script_dir, f'elements_{N_c}.input'), delimiter=',', dtype=int) - 1

# Call the function to preprocess the mesh from above
mesh_c = preprocess_mesh(coords_c, elements_c)

# 1. Using Jacobi Iterative solver
print("Running Jacobi Solver")
T_jac = np.zeros(mesh_c["num_elements"])
jac_residuals = []
for k in tqdm(range(20000), desc="Jacobi Progress"):
    T_old = T_jac.copy()
    T_new = np.zeros_like(T_jac)
    for i in range(mesh_c["num_elements"]):
        num_sum, den_sum = 0, 0
        for j in range(3):
            neighbor = mesh_c["neighbors"][i, j]
            edge_len = mesh_c["edge_lengths"][i, j]
            if neighbor != -1: # Internal edge
                dist = np.linalg.norm(mesh_c["circumcenters"][i] - mesh_c["circumcenters"][neighbor]) #delta value as distance between circumcenter of 2 triangles
                a_ij = edge_len / dist if dist > 1e-12 else 0
                num_sum += a_ij * T_old[neighbor]
            else: # Boundary edge
                v1_idx, v2_idx = mesh_c["elements"][i, j], mesh_c["elements"][i, (j + 1) % 3]
                midpoint = (mesh_c["coords"][v1_idx] + mesh_c["coords"][v2_idx]) / 2
                dist = np.linalg.norm(mesh_c["circumcenters"][i] - midpoint) #taking circumcnter of ith triangle with distance upto midpoint of boundary triangle edge
                a_ij = edge_len / dist if dist > 1e-12 else 0
            den_sum += a_ij
        source_term = mesh_c["areas"][i] #based on derivation done aleardy
        if den_sum > 1e-12:
            T_new[i] = (num_sum + source_term) / den_sum #as considered in derivation of problem 2(b)
    #find residual with difference between current and old value of T and in array get the maximum to compare with tolerence
    residual = np.linalg.norm(T_new - T_jac, ord=np.inf)
    jac_residuals.append(residual)
    T_jac = T_new.copy()
    if residual < 1e-6: #Tolerence=1e-6
        jac_iters = k + 1
        break

# 2. Using Gauss-Seidel Solver
print("Running Gauss-Seidel Solver")
T_gs = np.zeros(mesh_c["num_elements"])
gs_residuals = []
for k in tqdm(range(20000), desc="Gauss-Seidel Progress"):
    T_old = T_gs.copy()
    for i in range(mesh_c["num_elements"]):
        num_sum, den_sum = 0, 0
        for j in range(3):
            neighbor = mesh_c["neighbors"][i, j]
            edge_len = mesh_c["edge_lengths"][i, j]
            if neighbor != -1: # Internal edge
                dist = np.linalg.norm(mesh_c["circumcenters"][i] - mesh_c["circumcenters"][neighbor])
                a_ij = edge_len / dist if dist > 1e-12 else 0
                num_sum += a_ij * T_gs[neighbor] # Use updated T_gs value if known already
            else: # Boundary edge
                v1_idx, v2_idx = mesh_c["elements"][i, j], mesh_c["elements"][i, (j + 1) % 3]
                midpoint = (mesh_c["coords"][v1_idx] + mesh_c["coords"][v2_idx]) / 2
                dist = np.linalg.norm(mesh_c["circumcenters"][i] - midpoint)
                a_ij = edge_len / dist if dist > 1e-12 else 0
            den_sum += a_ij
        source_term = mesh_c["areas"][i]
        if den_sum > 1e-12:
            T_gs[i] = (num_sum + source_term) / den_sum
    #finding residual
    residual = np.linalg.norm(T_gs - T_old, ord=np.inf)
    gs_residuals.append(residual)
    if residual < 1e-6: #Tolerence=1e-6
        gs_iters = k + 1
        break

# 3. Using SOR Solver to find optimal alpha and solve
print("\nFinding optimal alpha for SOR")
best_alpha, best_iters = 1.0, gs_iters #best iteration is defined as number of iteration the gauss-seidel method took
for alpha_test in np.linspace(1.5, 1.95, 10): #optimal alpha cant be known in same way as problem 1 for structured mesh so giving range betweem 1.5 and 2 is over-relaxation that accelerates convergence.
    _, iters_test, _ = solve_fvm_sor(mesh_c, alpha=alpha_test, tol=1e-6, max_iter=gs_iters)
    if iters_test < best_iters:
        best_iters, best_alpha = iters_test, alpha_test
print(f"==> Found best alpha for N~40 to be approx {best_alpha:.3f}")

# Run final SOR with the best alpha
_, sor_iters, sor_residuals = solve_fvm_sor(mesh_c, alpha=best_alpha, tol=1e-6)

# Plotting for Problem 2(c)- Iterative solver comparison for N~40
plt.figure(figsize=(8, 6))
plt.semilogy(jac_residuals, label=f"Jacobi ({jac_iters} iters)") #this function considers y-axis (residual) at logarith scale and x-axis (iterations) at linear scale
plt.semilogy(gs_residuals, label=f"Gauss-Seidel ({gs_iters} iters)")
plt.semilogy(sor_residuals, label=f"SOR ($\\alpha$={best_alpha:.3f}, {sor_iters} iters)")
plt.title("Problem 2(c): Residual vs. Iteration (N~40)")
plt.xlabel("Iteration")
plt.ylabel("Infinity Norm of Solution Change- Residual")
plt.grid(True, which="both", linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "problem_2c_residuals_vs_iteration.png"))
print(f"Plot saved to {os.path.join(output_dir, 'problem_2c_residuals_vs_iteration.png')}")

# Problem 2(d): Color plots of the numerical solution for N~10,40,80 (gauss-seidel took lot of time and iterations so i chose SOR iterative method)
print("\nRunning Problem 2(d): Element Coloring Plots")
Ns_d = [10, 40, 80]
fig_d, axes_d = plt.subplots(1, len(Ns_d), figsize=(18, 5), sharey=True)
if len(Ns_d) == 1: axes_d = [axes_d]

for i, n in enumerate(Ns_d):
    print(f"Processing mesh N~{n} for coloring plot")
    # Load mesh files
    coords = np.loadtxt(os.path.join(script_dir, f'coordinates_{n}.input'), delimiter=',')
    elements = np.loadtxt(os.path.join(script_dir, f'elements_{n}.input'), delimiter=',', dtype=int) - 1
    
    mesh_d = preprocess_mesh(coords, elements)
    T_num, _, _ = solve_fvm_sor(mesh_d, alpha=1.8, tol=1e-6)
    # Coloured Plotting where each triangle corresponds to its own temperature calculated from SOR solver
    ax = axes_d[i]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=elements) #creating triangular object
    tpc = ax.tripcolor(triang, T_num, cmap='viridis', shading='flat') #draws triangles and color based on its corresponding temperature in T_num vector
    fig_d.colorbar(tpc, ax=ax, label="Temperature")
    ax.set_title(f"Numerical Solution (N~{n})")
    ax.set_xlabel("x")
    ax.set_aspect('equal', 'box')

axes_d[0].set_ylabel("y")
fig_d.suptitle("Problem 2(d): Temperature Distribution per Element", fontsize=16)
fig_d.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_d.savefig(os.path.join(output_dir, "problem_2d_coloring_map.png"))
print(f"Plot saved to {os.path.join(output_dir, 'problem_2d_coloring_map.png')}")

# Problem 2(e): Plot the temperature along the centerline y-axis y-axis T(0.5,y) for N~10, 20, 40
print("\nRunning Problem 2(e): Centerline Profile")
Ns_e = [10, 20, 40]
plt.figure(figsize=(8, 6))
y_fine = np.linspace(0.001, 0.999, 200)

for n in Ns_e:
    print(f"Processing mesh N~{n} for centerline plot")
    # Load files as before
    coords = np.loadtxt(os.path.join(script_dir, f'coordinates_{n}.input'), delimiter=',')
    elements = np.loadtxt(os.path.join(script_dir, f'elements_{n}.input'), delimiter=',', dtype=int) - 1
    
    mesh_e = preprocess_mesh(coords, elements)
    T_num, _, _ = solve_fvm_sor(mesh_e, alpha=1.8, tol=1e-6) # Using alpha = 1.8 (found to be best while trying different values)
    # Interpolate results onto the centerline
    T_centerline = griddata(mesh_e["circumcenters"], T_num, (0.5 * np.ones_like(y_fine), y_fine), method='linear')
    plt.plot(y_fine, T_centerline, '--', label=f"N~{n} (Numerical)")
# Plot analytical solution in same graph to compare with numerical solution from iterative solvers
T_ana_center = analytical_solution_p2(0.5, y_fine)
plt.plot(y_fine, T_ana_center, 'k-', linewidth=2, label="Analytical")
plt.title("Problem 2(e): Centerline Temperature at x=0.5")
plt.xlabel("y")
plt.ylabel("T(0.5, y)")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "problem_2e_centerline_analytical_numerical.png"))
print(f"Plot saved to {os.path.join(output_dir, 'problem_2e_centerline_analytical_numerical.png')}")

# Problem 2(f): Check spatial convergence rate (how the error changes as the mesh gets finer )
print("\nRunning Part 2(f): Spatial Convergence Rate")
Ns_f = [10, 20, 40, 80, 160]
errors, hs = [], []

for n in Ns_f:
    print(f"Processing mesh N~{n} for convergence plot")
    # Load coordinates and triangular element files
    coords = np.loadtxt(os.path.join(script_dir, f'coordinates_{n}.input'), delimiter=',')
    elements = np.loadtxt(os.path.join(script_dir, f'elements_{n}.input'), delimiter=',', dtype=int) - 1
    
    mesh_f = preprocess_mesh(coords, elements)
    T_num, _, _ = solve_fvm_sor(mesh_f, alpha=1.9, tol=1e-4, max_iter=2000) #since N=160 is also considered so increased large alpha to get convergence faster using SOR
    # Calculate error
    h = np.sqrt(np.mean(mesh_f["areas"]))
    hs.append(h)
    T_ana = analytical_solution_p2(mesh_f["circumcenters"][:, 0], mesh_f["circumcenters"][:, 1])
    err = np.sqrt(np.mean((T_num - T_ana)**2))
    errors.append(err)
    print(f"N~{n}, h={h:.4f}, RMS Error={err:.4e}")

# Plotting for Problem 2(f)
plt.figure(figsize=(8, 6))
plt.loglog(hs, errors, 'o-', label='RMS Error')
p_rate = np.log(errors[-2] / errors[-1]) / np.log(hs[-2] / hs[-1]) # Calculate the slope of the line on the log-log plot to show the convergence order
line_fit = np.exp(np.log(errors[-1]) - p_rate * np.log(hs[-1])) * np.array(hs)**p_rate
plt.loglog(hs, line_fit, 'r--', label=f'Order {p_rate:.2f} convergence')
plt.title("Problem 2(f): Spatial Convergence Rate (FVM)")
plt.xlabel("Characteristic Grid Spacing (h)")
plt.ylabel("Root Mean Square Error")
plt.grid(True, which="both", linestyle=':')
plt.gca().invert_xaxis() # Finer meshes (smaller h) are on the right
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "problem_2f_convergence.png"))
print(f"Plot saved to {os.path.join(output_dir, 'problem_2f_convergence.png')}")

# Show all the plots at the end
plt.show()


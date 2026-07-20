import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Physical Constants
# -----------------------------
kB = 1.380649e-23
T = 298.15
sigma = 0.010
An = 2.0e25
Vm = 1.0e-27
MW_lipid = 0.600
rho_lipid = 1000
kg = 12
ENABLE_NUCLEATION = True

kBT = kB * T

# -----------------------------
# 2. Solvent pure properties
# -----------------------------
MW_water, MW_ethanol = 0.018015, 0.04607
rho_water, rho_ethanol = 1000, 789


# Solubility model
def solubility_mix(v_water, v_ethanol):
    x_water_pure, x_ethanol_pure = 1e-8, 5e-3
    ln_x = v_water*np.log(x_water_pure) + v_ethanol*np.log(x_ethanol_pure)
    return np.exp(ln_x)

FRR_values = [ 3,4,5,6,7,8, 9]
C_eth_phase_values = [2,4,6,8, 10]

# -----------------------------
# Helper: build mixture and C0 from FRR and concentration multiplier
# -----------------------------
def mixture_from_FRR(FRR, C_eth_phase):
    # Volume fractions that sum to 1
    v_ethanol = 1.0 / (FRR + 1.0)
    v_water = FRR / (FRR + 1.0)

    # Mass fractions
    m_eth = v_ethanol * rho_ethanol
    m_wat = v_water * rho_water
    w_eth = m_eth / (m_eth + m_wat)
    w_wat = m_wat / (m_eth + m_wat)

    # Mixture density
    rho_mix = 1.0 / (w_eth/rho_ethanol + w_wat/rho_water)

    # Mole fraction of ethanol in solvent
    x_eth_solv = (w_eth/MW_ethanol) / (w_eth/MW_ethanol + w_wat/MW_water)
    M_solvent_mix = x_eth_solv*MW_ethanol + (1.0 - x_eth_solv)*MW_water

    # Solubility in the mixture
    Csat_inf = solubility_mix(v_water, v_ethanol) * (rho_mix/M_solvent_mix) * MW_lipid

    # Initial dissolved concentration after mixing (with multiplier)
    C0 = C_eth_phase * v_ethanol 

    return {
        'FRR': FRR,
        'v_ethanol': v_ethanol,
        'v_water': v_water,
        'rho_mix': rho_mix,
        'M_solvent_mix': M_solvent_mix,
        'Csat_inf': Csat_inf,
        'C0': C0
    }

# -----------------------------
# 3. z-average diameter calculation
# -----------------------------
def z_average_diameter(L, nL):
    num = np.trapz(nL * L**7, L)
    den = np.trapz(nL * L**6, L)
    if den <= 0:
        return np.nan
    return num / den

# -----------------------------
# 4. Simulation function
# -----------------------------
def run_simulation(mix, N=2000, t_span=(1e-6, 1.0), t_eval=None):
    if t_eval is None:
        t_eval = np.logspace(-6, 0, 500)

    Csat_inf = mix['Csat_inf']
    C0 = mix['C0']

    # Size grid
    L_min, L_max = 1e-9, 1000e-9
    L = np.linspace(L_min, L_max, N)
    dL = L[1] - L[0]

    # Volumes
    V_L = (np.pi/6) * L**3

    # Initial conditions
    n0 = np.zeros(N)
    y0 = np.concatenate([n0, [C0]])

    step_count = [0]

    def compute_birth_kernel(L, Lc, J, dL):
        if J <= 0:
            return np.zeros_like(L)
        Lc_safe = np.clip(Lc, L[0] + 10*dL, L[-1] - 10*dL)
        sigma_nuc = max(0.2 * Lc_safe, 5*dL)
        birth = np.exp(-((L - Lc_safe)**2) / (2*sigma_nuc**2))
        norm = np.trapz(birth, L)
        if norm <= 1e-50:
            return np.zeros_like(L)
        return J * birth / norm

    def pbe_rhs(t, y):
        step_count[0] += 1

        n = np.maximum(y[:-1].copy(), 0.0)
        C = max(y[-1], 1e-20)
        S = C / Csat_inf

        # Nucleation (CNT)
        if ENABLE_NUCLEATION and S > 1.001:
            lnS = np.log(S)
            deltaG_crit = (16*np.pi*sigma**3*Vm**2) / (3*(kBT*lnS)**2)
            deltaG_norm = deltaG_crit / kBT
            J_nuc = 0.0 if deltaG_norm > 60 else An * np.exp(-deltaG_norm)
            Lc_nuc = (4*sigma*Vm) / (kBT*lnS)
            Lc_nuc = np.clip(Lc_nuc, L_min + 10*dL, L_max - 10*dL)
        else:
            J_nuc = 0.0
            Lc_nuc = L_min

        Bn = compute_birth_kernel(L, Lc_nuc, J_nuc, dL)

        # Growth with Kelvin effect
        Sk = np.exp((4*sigma*Vm) / (kBT * L))
        G = Csat_inf * kg * L * (S - Sk)

        # Upwind transport
        G_face = 0.5 * (G[:-1] + G[1:])
        n_face = np.where(G_face >= 0, n[:-1], n[1:])
        flux = G_face * n_face

        flux_left = min(G[0], 0) * n[0]
        flux_right = max(G[-1], 0) * n[-1]

        dndt = np.zeros(N)
        dndt[0] = Bn[0] - (flux[0] - flux_left) / dL
        dndt[1:-1] = Bn[1:-1] - (flux[1:] - flux[:-1]) / dL
        dndt[-1] = Bn[-1] - (flux_right - flux[-2]) / dL

        # Mass balance
        dVtot_dt = np.trapz(dndt * V_L, L)
        dCdt = -rho_lipid * dVtot_dt

        return np.concatenate([dndt, [dCdt]])

    sol = solve_ivp(
        pbe_rhs, t_span, y0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-5,
        atol=1e-10,
        max_step=1e-2
    )

    t = sol.t
    n_hist = sol.y[:-1, :]
    C_hist = sol.y[-1, :]

    # Compute z-average diameter vs time
    Dz_nm = np.array([z_average_diameter(L, n_hist[:, i]) for i in range(len(t))]) * 1e9

    return {
        'mix': mix,
        'L': L,
        't': t,
        'n_hist': n_hist,
        'C_hist': C_hist,
        'Dz_nm': Dz_nm
    }

# -----------------------------
# 5. Grid of FRR and concentration multipliers
# -----------------------------


# Storage for results
results = {}

print("\n" + "="*80)
print("STARTING GRID SIMULATION: FRR x Concentration")
print("="*80)

for FRR in FRR_values:
    for C_eth_phase in C_eth_phase_values:
        mix = mixture_from_FRR(FRR, C_eth_phase)
        print(f"Running: FRR={FRR}, C_eth={C_eth_phase:.1f}, C0={mix['C0']:.3f} kg/m³", end=" ... ")
        
        res = run_simulation(mix, N=1000)
        results[(FRR, C_eth_phase)] = res
        
        print(f"Done. Final Dz: {res['Dz_nm'][-1]:.2f} nm")

print("="*80)

# -----------------------------
# 6. Save z-average diameter vs TIME data to file
# -----------------------------
# Save full time series as numpy array
data_dict = {}
for key, res in results.items():
    FRR, C_eth_phase = key
    data_dict[f'FRR{FRR}_Conc{C_eth_phase:.1f}_time'] = res['t']
    data_dict[f'FRR{FRR}_Conc{C_eth_phase:.1f}_Dz_nm'] = res['Dz_nm']

np.savez('Dz_vs_time_grid1.npz', **data_dict)
print("Full time series data saved to 'Dz_vs_time_grid1.npz'")

# Also save as CSV files for each FRR-Concentration combination
import os
os.makedirs('Dz_time_series', exist_ok=True)

for key, res in results.items():
    FRR, C_eth_phase = key
    filename = f'Dz_time_series/FRR{FRR}_Conc{C_eth_phase:.1f}.csv'
    data = np.column_stack([res['t'], res['Dz_nm']])
    np.savetxt(filename, data, delimiter=',', 
               header='Time(s),Dz(nm)', comments='')

print("Individual CSV files saved in 'Dz_time_series/' folder")

# Create a summary text file
with open('Dz_data_summary1.txt', 'w') as f:
    f.write("FRR\tC_eth_phase\tC0(kg/m3)\tFinal_Dz(nm)\n")
    for key, res in results.items():
        FRR, C_eth_phase = key
        C0 = res['mix']['C0']
        final_Dz = res['Dz_nm'][-1]
        f.write(f"{FRR}\t{C_eth_phase:.1f}\t{C0:.4f}\t{final_Dz:.4f}\n")
print("Summary saved to 'Dz_data_summary1.txt'")

# -----------------------------
# 7. Plot: Dz vs time for each FRR (separate subplots)
# -----------------------------

FRR_values1 = [ 3,9]
C_eth_phase_values1 = [2,10]

fig, axes = plt.subplots(len(FRR_values1), 1, figsize=(6, 3*len(FRR_values1)))
if len(FRR_values1) == 1:
    axes = [axes]

colors = plt.cm.viridis(np.linspace(0, .9, len(C_eth_phase_values)))

for idx, FRR in enumerate(FRR_values1):
    ax = axes[idx]
    for jdx, C_eth_phase in enumerate(C_eth_phase_values):
        res = results[(FRR, C_eth_phase)]
        C0 = res['mix']['C0']
        ax.semilogx(res['t'], res['Dz_nm'], lw=2, color=colors[jdx], 
                   label=f'Lipid concentration ={C_eth_phase:.1f} ($C_{{mix}}$={C0:.2f})')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Dz (nm)', fontsize=10)
    ax.set_xlim(1e-5, None)
    ax.set_title(f'z-average diameter vs Time for FRR={FRR}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('Dz_vs_time_FRR_grid1.png', dpi=150)
plt.savefig('Dz_vs_time_FRR_grid1.pdf', dpi=150, bbox_inches='tight')  # Save as PDF
plt.show()


# -----------------------------
# 8. Plot: Dz vs time for each Conc (separate subplots)
# -----------------------------
fig, axes = plt.subplots(len(C_eth_phase_values1), 1, figsize=(6, 3*len(C_eth_phase_values1)))
if len(C_eth_phase_values) == 1:
    axes = [axes]


colors = plt.cm.plasma(np.linspace(0, .9, len(FRR_values)))

for idx, C_eth_phase in enumerate(C_eth_phase_values1):
    ax = axes[idx]
    for jdx, FRR in enumerate(FRR_values):
        res = results[(FRR, C_eth_phase)]
        C0 = res['mix']['C0']
        ax.semilogx(res['t'], res['Dz_nm'], lw=2, color=colors[jdx], 
                   label=f'FRR={FRR} ($C_{{mix}}$={C0:.2f})')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Dz (nm)', fontsize=10)
    ax.set_xlim(1e-5, None)
    ax.set_title(f'z-average diameter vs Time for lipid concentration={C_eth_phase:.1f}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('Dz_vs_time_Conc_grid1.png', dpi=150)
plt.savefig('Dz_vs_time_Conc_grid1.pdf', dpi=150, bbox_inches='tight')  # Save as PDF
plt.show()

# -----------------------------
# 9. Heatmap: Final Dz as function of FRR and Conc
# -----------------------------
Dz_grid = np.zeros((len(FRR_values), len(C_eth_phase_values)))
for i, FRR in enumerate(FRR_values):
    for j, C_eth_phase in enumerate(C_eth_phase_values):
        Dz_grid[i, j] = results[(FRR, C_eth_phase)]['Dz_nm'][-1]

plt.figure(figsize=(10, 6))
im = plt.imshow(Dz_grid, aspect='auto', origin='lower', cmap='viridis',
                extent=[C_eth_phase_values[0], C_eth_phase_values[-1], 
                       FRR_values[0], FRR_values[-1]])
plt.colorbar(im, label='Final Dz (nm)')
plt.xlabel('C_eth_phase (kg/m³)')
plt.ylabel('FRR')
plt.title('Final z-average Diameter (nm) - Heatmap')
plt.xticks(C_eth_phase_values)
plt.yticks(FRR_values)
plt.tight_layout()
plt.savefig('Dz_final_heatmap1.png', dpi=150)
plt.show()

# -----------------------------
# 10. 3D surface plot: Final Dz vs FRR and Conc
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D

FRR_mesh, Conc_mesh = np.meshgrid(FRR_values, C_eth_phase_values)
Dz_mesh = Dz_grid.T  # transpose for meshgrid orientation

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(FRR_mesh, Conc_mesh, Dz_mesh, cmap='viridis', 
                       edgecolor='none', alpha=0.9)
ax.set_xlabel('FRR')
ax.set_ylabel('C_eth_phase (kg/m³)')
ax.set_zlabel('Final Dz (nm)')
ax.set_title('Final z-average Diameter as function of FRR and Concentration')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('Dz_final_3D_surface1.png', dpi=150)
plt.show()

print("\nAll plots and time series data saved successfully!")
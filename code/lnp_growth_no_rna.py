#### LNP growth without RNA, THP, 2026 ###
#### Lipid concentrations: 10 mM and 5 mM ###

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# -----------------------------
# Physical Constants & Parameters
# -----------------------------
Aham = 0.7         # Hamaker constant [kBT]
RgPEG = 3.6          # PEG Flory radius [nm]
fPEG = 0.015         # PEG volume fraction
vL = 3.0             # Lipid volume [nm^3]
eta = 1e-3           # Solvent viscosity [Pa·s]
T = 298              # Temperature [K]
kB = 1.38e-23        # Boltzmann constant [J/K]
e0 = 1.602e-19       # Elementary charge [C]
lb = 0.71            # Bjerrum length [nm]
csalt = 0.025        # Salt concentration [mol/L] = 25 mM
R0 = 3.0             # Initial LNP radius [nm]

# Lipid concentrations to test
lipid_concentrations = {
    "10 mg/ml lipid": 0.010,   # mg/L
    "5 mg/ml lipid": 0.005     # mg/L
}

# -----------------------------
# Load interpolated psi(R) data from CSV
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(BASE_DIR, "data", "psi_vs_R_25mM_noRNA.csv")

df = pd.read_csv(data_file)

interp_psi_mV = interp1d(
    df["R_nm"],
    df["psi_mV"],
    kind="linear",
    fill_value="extrapolate"
)

# -----------------------------
# Derived Parameters independent of cL
# -----------------------------
K2 = 2 * np.pi * RgPEG * fPEG / (3 * vL)              # [1/nm^2]
lD = 1 / np.sqrt(8 * np.pi * lb * 0.60223 * csalt)   # Debye length [nm]

print(f"K2 = {K2:.4e} 1/nm^2")
print(f"Debye length = {lD:.4f} nm")

# -----------------------------
# DLVO Potential: vdW + Electrostatics
# -----------------------------
def Wdlvo(D, R, psi, lD):
    vdW = -Aham * R / (12 * D)

    electrostatic = (
        R
        * psi**2
        * np.log(1 + np.exp(-D / lD))
        / 4
        / 4
        * (80 / 52)
    )

    return vdW + electrostatic


def Wb_dlvo(R, psi, lD):
    result = minimize_scalar(
        lambda D: -Wdlvo(D, R, psi, lD),
        bounds=(0.2, 10),
        method="bounded"
    )

    if result.success:
        return Wdlvo(result.x, R, psi, lD)
    else:
        return np.nan

# -----------------------------
# Time grid
# -----------------------------
t_eval = np.logspace(-9, 0, 500)  # 1 ns to 1 s

# -----------------------------
# Solve for each lipid concentration
# -----------------------------
all_results = {}

plt.figure(figsize=(8, 5))

for label, cL in lipid_concentrations.items():

    # K1 depends on lipid concentration
    K1 = vL * cL * kB * T / (np.pi * 3 * eta) * 1e27  # [nm^3/s]

    print(f"{label}: cL = {cL:.4f} mol/L, K1 = {K1:.4e} nm^3/s")

    # -----------------------------
    # Radius Growth ODE
    # -----------------------------
    def dRdt(t, y):
        R = y[0]

        psi_mV = float(interp_psi_mV(R))
        psi = psi_mV * 1e-3 * e0 / (kB * T)  # dimensionless surface potential

        Wb = Wb_dlvo(R, psi, lD)

        dR = K1 * np.exp(-K2 * R**2 - Wb) / R**2

        return [dR]

    # -----------------------------
    # Solve ODE
    # -----------------------------
    sol = solve_ivp(
        dRdt,
        [1e-9, 1e6],
        [R0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        print(f"Solver failed for {label}: {sol.message}")

    all_results[label] = sol

    # Plot
    plt.plot(
        sol.t,
        sol.y[0],
        linewidth=2,
        label=label
    )

    # Save individual CSV
    output_df = pd.DataFrame({
        "time_s": sol.t,
        "radius_nm": sol.y[0]
    })

    filename = f"lnp_radius_vs_time_{label.replace(' ', '_').replace('mM', 'mM')}.csv"
    output_df.to_csv(filename, index=False)

    print(f" Saved {label} result to '{filename}'")

# -----------------------------
# Plot formatting
# -----------------------------
plt.xscale("log")
plt.xlabel("Time (s)")
plt.ylabel("LNP Radius R(t) [nm]")
plt.title("LNP Growth without RNA: 5 mM vs 10 mM Lipid")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Save combined CSV
# -----------------------------
combined_df = pd.DataFrame({
    "time_s": t_eval,
    "radius_nm_10mM": all_results["10 mM lipid"].y[0],
    "radius_nm_5mM": all_results["5 mM lipid"].y[0]
})

combined_df.to_csv("lnp_radius_vs_time_5mM_10mM.csv", index=False)

print(" Saved combined data to 'lnp_radius_vs_time_5mM_10mM.csv'")
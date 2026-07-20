import numpy as np
import pandas as pd

# Parameters
D_PEI = 1e-5  # cm^2/s

# Flow rates (mL/min)
Q_values = [20, 30, 40, 60, 80]

rows = []

print("Flow rate (mL/min) | tau_M (ms) | tau_M (s) | Kolmogorov length (µm)")

for Q in Q_values:
    # Mixing time from empirical correlation (in ms)  ## ACS Nano 2019
    tau_M_ms = 1.266e3 * (Q ** -1.478)
    
    # Convert to seconds
    tau_M_s = tau_M_ms * 1e-3
    
    # Kolmogorov length scale eta (cm)
    eta_cm = 2 * np.sqrt(tau_M_s * D_PEI)
    
    # Convert to µm
    eta_um = eta_cm * 1e4  # 1 cm = 10^4 µm

    #prepare output csv
    rows.append({
        "Q_ml_min": Q,
        "Kolmogorov_length_um": eta_um
    })
    
    print(f"{Q:>15} | {tau_M_ms:10.3f} | {tau_M_s:10.3e} | {eta_um:10.3f}")

df = pd.DataFrame(rows)
df.to_csv("flow_rate_lengthscale.csv", index=False)
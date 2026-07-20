#### CODE FOR FIGURE 3. thp 7-16-26####

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
k_B = 1.38e-23      # J/K
T = 298             # K
eta = 1e-3          # Pa.s

D_mRNA_um2_per_s = 1.5e-11 * 1e12   # µm²/s, 1.5 for 2000 nt, 2.3 for 1000 nt;

# Target mixing lengths
L_targets = [3.9, 2.9, 2.3]   # µm

# -----------------------------
# Load LNP radius vs time file
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(BASE_DIR, "data", "lnp_radius_vs_time_5mM_10mM.csv")
df = pd.read_csv(data_file)
df = df.sort_values("time_s").dropna().copy()

# -----------------------------
# Function to calculate cumulative mixing lengths
# -----------------------------
def calculate_mixing_lengths(df, radius_col):
    times = []
    radii_nm = []
    L_lnp_list = []
    L_mrna_list = []
    L_combined_list = []

    variance_lnp_total = 0.0
    variance_mrna_total = 0.0
    previous_time = 0.0

    for _, row in df.iterrows():
        current_time = float(row["time_s"])
        radius_nm = float(row[radius_col])

        dt = current_time - previous_time

        if dt > 0:
            radius_m = radius_nm * 1e-9

            # Stokes-Einstein diffusivity for LNP
            D_LNP_m2_s = k_B * T / (6.0 * np.pi * eta * radius_m)

            # Convert m²/s to µm²/s
            D_LNP_um2_s = D_LNP_m2_s * 1e12

            # Variance accumulation
            variance_lnp_total += 2.0 * D_LNP_um2_s * dt
            variance_mrna_total += 2.0 * D_mRNA_um2_per_s * dt

        L_lnp = np.sqrt(variance_lnp_total)
        L_mrna = np.sqrt(variance_mrna_total)
        L_combined = np.sqrt(variance_lnp_total + variance_mrna_total)

        times.append(current_time)
        radii_nm.append(radius_nm)
        L_lnp_list.append(L_lnp)
        L_mrna_list.append(L_mrna)
        L_combined_list.append(L_combined)

        previous_time = current_time

    result_df = pd.DataFrame({
        "time_s": times,
        "radius_nm": radii_nm,
        "L_lnp_um": L_lnp_list,
        "L_mRNA_um": L_mrna_list,
        "L_combined_um": L_combined_list
    })

    return result_df

# -----------------------------
# Function to find target crossing
# -----------------------------
def find_target_point(mix_df, L_target):
    reached_df = mix_df[mix_df["L_combined_um"] >= L_target]

    if len(reached_df) == 0:
        return np.nan, np.nan, np.nan

    row = reached_df.iloc[0]

    target_time = row["time_s"]
    target_radius_nm = row["radius_nm"]
    target_L = row["L_combined_um"]

    return target_time, target_radius_nm, target_L

# -----------------------------
# Analyze 10 mM and 5 mM
# -----------------------------
cases = {
    "10 mM": "radius_nm_10mM",
    "5 mM": "radius_nm_5mM"
}

summary_rows = []

for label, radius_col in cases.items():

    mix_df = calculate_mixing_lengths(df, radius_col)

    # Save full mixing length data
    safe_label = label.replace(" ", "")
    mix_csv = f"mixing_lengths_{safe_label}.csv"
    mix_df.to_csv(mix_csv, index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(6.0, 4.8))
    plt.xlim(-0.05, 0.50)
    plt.ylim(-0.4, 6.0)

    plt.plot(
        mix_df["time_s"],
        mix_df["L_lnp_um"],
        color="blue",
        linewidth=1.3,
        label="LNP Mixing Length"
    )

    plt.plot(
        mix_df["time_s"],
        mix_df["L_mRNA_um"],
        color="green",
        linewidth=1.3,
        label="mRNA Mixing Length"
    )

    plt.plot(
        mix_df["time_s"],
        mix_df["L_combined_um"],
        color="red",
        linestyle="-.",
        linewidth=1.3,
        label="Combined Mixing Length"
    )

    # Plot target lines and target points
    target_colors = ["black", "gray", "purple"]

    for L_target, target_color in zip(L_targets, target_colors):

        target_time, target_radius_nm, target_L = find_target_point(mix_df, L_target)

        # Save summary for this target
        summary_rows.append({
            "lipid_concentration": label,
            "target_length_um": L_target,
            "target_time_s": target_time,
            "radius_at_mixing_time_nm": target_radius_nm,
            "target_combined_length_um": target_L
        })

        # Target horizontal line
        plt.axhline(
            L_target,
            color=target_color,
            linestyle=":",
            linewidth=3.2,
            label=f"Target Length = {L_target} µm"
        )

        # Target crossing point
        if not np.isnan(target_time):
            plt.scatter(
                target_time,
                target_L,
                color="red",
                s=60,
                zorder=5
            )

            #plt.text(
            #    target_time,
            #    target_L,
            #    f"  {L_target} µm\n  t={target_time:.4g}s",
            #    fontsize=12,
            #    verticalalignment="bottom"
            #)

    plt.xlabel("Time, $t$ (s)", fontsize=12)
    plt.ylabel("Mixing Length (µm)", fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plot_file = f"mixing_length_plot_{safe_label}.png"

    plot_file = f"mixing_length_plot_{safe_label}.pdf"
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    plt.show()

# -----------------------------
# Save summary CSV
# -----------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("mixing_target_summary_5mM_10mM.csv", index=False)

print("\nSummary:")
print(summary_df)

import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import time
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))

# import simulation
from LNP_KMC import run_simulation


# =========================
# HELPER FUNCTION
# =========================
def lipid_string_to_number_mM(lipid_string):
    """
    Converts lipid concentration string to numeric mM.

    Examples:
        '10 mM' -> 10.0
        '5 mM'  -> 5.0
    """
    return float(re.findall(r"[-+]?\d*\.\d+|\d+", str(lipid_string))[0])


# =========================
# TASK FUNCTION
# =========================
def task(params):
    try:
        FRR, conc_csv, R0, run_id = params

        cL_ethanol = conc_csv 

        result = run_simulation(
            FRR=FRR,
            cL_ethanol=cL_ethanol,
            R0=R0,
            try_id=run_id,
            seed=1000 + run_id
        )

        return result

    except Exception as e:
        print(f"[ERROR] FRR={FRR}, conc={conc_csv}, R0={R0}, run={run_id}: {e}")
        return None


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    start = time.time()

    # -------------------------
    # LOAD CSV
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(BASE_DIR, "data", "mixing_target_summary_5mM_10mM.csv")
    df = pd.read_csv(input_csv)

    # FRR = 3 for all rows
    FRR_values = [3] * len(df)

    conc_values = df["lipid_concentration"].apply(lipid_string_to_number_mM).values

    # R0 from radius_at_mixing_time_nm
    R0_values = df["radius_at_mixing_time_nm"].values

    input_df = pd.DataFrame({
        "FRR": FRR_values,
        "Concentration": conc_values,
        "R0": R0_values
    }).dropna()

    n_repeats = 50

    all_jobs = [
        (FRR, conc, R0, i + 1)
        for FRR, conc, R0 in zip(
            input_df["FRR"].values,
            input_df["Concentration"].values,
            input_df["R0"].values
        )
        for i in range(n_repeats)
    ]

    print(f"Loaded input CSV: {input_csv}")
    print(f"Unique conditions: {len(input_df)}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total jobs: {len(all_jobs)}")

    print("\nInput conditions used:")
    print(input_df.to_string(index=False))

    # -------------------------
    # SLURM CORE DETECTION
    # -------------------------
    n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count()))
    print(f"\n Using {n_cores} cores")

    # -------------------------
    # RUN PARALLEL
    # -------------------------
    with Pool(processes=n_cores) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(task, all_jobs),
                total=len(all_jobs)
            )
        )

    # -------------------------
    # RESULTS
    # -------------------------
    results_clean = [r for r in results if r is not None]

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    output_file = os.path.join(out_dir, "kmc_results.csv")

    df_out = pd.DataFrame(results_clean)
    df_out.to_csv(output_file, index=False)

    print(f"\nSaved {len(results_clean)} results to {output_file}")
    print(f" Time: {(time.time() - start) / 60:.2f} minutes")
import numpy as np
import math
from random import choices, expovariate
import os


def run_simulation(FRR, cL_ethanol, R0, N1=2000, try_id=0, seed=0):

    np.random.seed(seed)

    # --- Dielectric properties from FRR (MUST BE CALCULATED FIRST) ---
    water_frac = FRR / (FRR + 1)
    epsilon_water = 78.5
    epsilon_ethanol = 24.3
    epsilon_r = water_frac * epsilon_water + (1 - water_frac) * epsilon_ethanol
    lB = 0.71 * (78.5 / epsilon_r)  # Bjerrum length scales with dielectric
    dielectric_scaling = 78.5 / epsilon_r  # For scaling surface potentials

    # --- Initial ---
    cL = cL_ethanol / (FRR + 1)
    NP_ratio = 6

    RNA_nt = 2000  # RNA size
    c_RNA_mol_per_L = cL / NP_ratio / RNA_nt / 2  # mol/L

    vL = 3.0
    fw = 0.2
    NA = 6.022e23  # Avogadro's number
    c0 = cL * NA * vL / (4.0 * math.pi * (R0)**3) / (1.0 - fw)
    N_init_LNP = 2000
    vol_system = N_init_LNP / c0
    N_RNA = int(round(c_RNA_mol_per_L * vol_system * NA))

    tot_LNPs = N_init_LNP
    lnp_with_rna = 0
    free_rna = N_RNA

    # --- PARAMETERS ---
    tot_particles = tot_LNPs + free_rna + lnp_with_rna
    f_with = lnp_with_rna / tot_particles
    free_rna_per_lnp = free_rna / N_init_LNP

    N0=round(N1*(1+free_rna_per_lnp))
    N_RNA_LNP = round(f_with * N0)
    N_free_RNA = round(free_rna_per_lnp * N1)
    N_empty_LNP = N0 - N_RNA_LNP - N_free_RNA
    N_init_LNP = N_empty_LNP + N_RNA_LNP

    fiPEG0 = 0.015 * 0.9
    R_RNA = 9.0

    eta = 0.001
    vRNA = 601.0
    RF = 3.6
    Aham = 0.7
    temperature = 298.0
    csalt = 0.025

    # Debye length (depends on lB which depends on epsilon_r)
    lD = 1.0 / np.sqrt(8.0 * math.pi * lB * 0.60223 * csalt)

    # charge / composition derived params
    rhoRNA = -2000.0 / vRNA
    rhoL = 1.0 / vL
    gfiRNAnp1 = 5.0 * rhoL / (5.0 * rhoL - 6.0 * rhoRNA)
    gELON = 1  # electrostatics on initially

    c0 = cL * 0.6022 * 3.0 * vL / (4.0 * math.pi * R0**3) / (1.0 - fw)
    vol_system = N_init_LNP / c0
    
    # Create output directory
    output_dir = f"results_RO{R0}_cL{cL_ethanol}"
    os.makedirs(output_dir, exist_ok=True)

    # Print diagnostic information
    print(f"\n{'='*70}")
    print(f"SIMULATION SETUP")
    print(f"{'='*70}")
    print(f"FRR (H₂O:EtOH): {FRR}:1")
    print(f"Water fraction: {water_frac*100:.1f}%")
    print(f"Dielectric constant (ε): {epsilon_r:.2f}")
    print(f"Bjerrum length (lB): {lB:.4f} nm (reference: 0.71 nm in water)")
    print(f"Debye length (lD): {lD:.4f} nm")
    print(f"Dielectric scaling factor: {dielectric_scaling:.3f}")
    print(f"cL (ethanol): {cL_ethanol} mM")
    print(f"cL (aqueous): {cL:.4f} mM")
    print(f"Salt: {csalt} M")
    print(f"Particles: Empty={N_empty_LNP}, RNA-LNP={N_RNA_LNP}, Free RNA={N_free_RNA}")
    print(f"System volume: {vol_system:.2e} nm³")
    print(f"{'='*70}\n")

    np.random.seed(seed)

    # Array of particle properties
    lnpa = np.zeros((N0, 6), dtype=float)
    lnpa[:, :] = -1.0

    # --- Fusion rules ---
    def fusion_result(t1, t2):
        if t1 == 2 and t2 == 2:
            return False, None
        if t1 == 0 and t2 == 0:
            return True, 0
        return True, 1

    # --- Core functions ---
    def get_psi(lnpa_arr, i):
        """
        Compute surface potential with DIELECTRIC SCALING.
        Surface potential scales inversely with dielectric constant.
        """
        ptype = int(lnpa_arr[i, 4])
        psi = 0.0
        
        if ptype == 0:
            # Empty LNP: fiRNA = 0
            psi_base = -130.0 * np.tanh(13.0 * np.tanh(lnpa_arr[i, 0] / 16.0) * (0.0 - gfiRNAnp1))
            psi = psi_base * dielectric_scaling  # Scale by dielectric
            
        elif ptype == 1:
            # LNP with RNA
            nRNA_particle = lnpa_arr[i, 5]
            fiRNA_particle = 0.0
            if lnpa_arr[i, 0] > 0:
                fiRNA_particle = nRNA_particle * vRNA / (4.0 / 3.0 * math.pi * lnpa_arr[i, 0]**3 * (1.0 - fw))
            psi_base = -130.0 * np.tanh(13.0 * np.tanh(lnpa_arr[i, 0] / 16.0) * (fiRNA_particle - gfiRNAnp1))
            psi = psi_base * dielectric_scaling  # Scale by dielectric
            
        elif ptype == 2:
            # Free RNA
            psi = 0.0
        
        lnpa_arr[i, 3] = psi * gELON
        return lnpa_arr[i, 3]

    def mW(D, n1, n2):
        """
        Interaction potential with CORRECT dielectric constant.
        """
        r1 = lnpa[n1, 0]
        r2 = lnpa[n2, 0]
        psi1 = lnpa[n1, 3]
        psi2 = lnpa[n2, 3]

        # Prefactor now uses actual epsilon_r (not hardcoded 80)
        pref = 0.85 * epsilon_r * 8.85e-4 / (1.38 * temperature * 4.0)
        
        D = np.array(D, dtype=float)
        denom_rad = (r1 + r2) if (r1 + r2) != 0 else 1e-12

        W_vdW = -Aham / 6.0 * r1 * r2 / denom_rad / D
        
        psi_sq_sum = psi1**2 + psi2**2
        if psi_sq_sum == 0:
            W_elec = 0.0
        else:
            term1 = 2.0 * psi1 * psi2 / psi_sq_sum
            logterm = np.log((1.0 + np.exp(-D / lD)) / (1.0 - np.exp(-D / lD)))
            W_elec = pref * r1 * r2 / denom_rad * psi_sq_sum * (term1 * logterm + np.log(1.0 - np.exp(-2.0 * D / lD)))
        
        W = W_vdW + W_elec
        return -W

    def get_rate1(lnpa_arr, n1, nmax, rate_out):
        """
        Calculate rates with safety checks.
        """
        type1 = int(lnpa_arr[n1, 4])
        rate_out[:nmax] = 0.0
        
        MAX_BARRIER = 50.0  # Cap barrier for numerical stability

        for j in range(nmax):
            if j == n1:
                rate_out[j] = 0.0
                continue
            
            type2 = int(lnpa_arr[j, 4])
            allowed, _ = fusion_result(type1, type2)
            if not allowed:
                rate_out[j] = 0.0
                continue

            # RNA interactions (simplified)
            if (type1 == 2 and type2 in [0, 1]) or (type1 in [0, 1] and type2 == 2):
                cfactor = 1.38e-23 * temperature * 1e27
                rate_out[j] = cfactor * 2.0 / (3.0 * eta)
                continue

            # LNP-LNP interactions
            r1 = lnpa_arr[n1, 0]
            r2 = lnpa_arr[j, 0]
            
            if r1 <= 0 or r2 <= 0:
                rate_out[j] = 0.0
                continue
            
            denom_rad = max(r1 + r2, 1e-12)
            Epeg = (2.0 * math.pi * RF / (3.0 * vL * (1.0 - fw)) *
                    r1 * r2 * (lnpa_arr[n1, 1] * r1 + lnpa_arr[j, 1] * r2) / denom_rad)
            
            Wbmax = 0.0
            if abs(lnpa_arr[n1, 3]) > 5.0 and abs(lnpa_arr[j, 3]) > 5.0 and lnpa_arr[n1, 3] * lnpa_arr[j, 3] > 200.0:
                try:
                    Ds = np.linspace(0.1, 2.0, num=50)
                    Wvals = mW(Ds, n1, j)
                    if np.all(np.isfinite(Wvals)):
                        Wbmax = -np.min(Wvals)
                except:
                    Wbmax = 0.0
            
            total_barrier = min(Epeg + Wbmax, MAX_BARRIER)
            
            cfactor = 1.38e-23 * temperature * 1e27
            try:
                rate_out[j] = cfactor * 2.0 / (3.0 * eta) * (r1 + r2)**2 / (r1 * r2) * math.exp(-total_barrier)
            except:
                rate_out[j] = 0.0

    def merge(lnpa_arr, n1, n2, nmax):
        nm, nd = (n1, n2) if n1 < n2 else (n2, n1)
        type1 = int(lnpa_arr[n1, 4])
        type2 = int(lnpa_arr[n2, 4])
        allowed, new_type = fusion_result(type1, type2)
        if not allowed:
            return

        rnew = (lnpa_arr[n1, 0]**3 + lnpa_arr[n2, 0]**3)**(1.0/3.0)
        vol_sum = max((lnpa_arr[n1, 0]**3 + lnpa_arr[n2, 0]**3), 1e-12)
        fiPEGnew = (lnpa_arr[n1, 1] * lnpa_arr[n1, 0]**3 + lnpa_arr[n2, 1] * lnpa_arr[n2, 0]**3) / vol_sum
        nRNA_new = lnpa_arr[n1, 5] + lnpa_arr[n2, 5]
        fiRNA_new = 0.0
        if new_type == 1:
            fiRNA_new = nRNA_new * vRNA / (4.0/3.0 * math.pi * rnew**3 * (1.0 - fw))

        lnpa_arr[nm, 0] = rnew
        lnpa_arr[nm, 1] = fiPEGnew
        lnpa_arr[nm, 2] = fiRNA_new
        lnpa_arr[nm, 3] = 0.0
        lnpa_arr[nm, 4] = float(new_type)
        lnpa_arr[nm, 5] = nRNA_new
        get_psi(lnpa_arr, nm)

        if nd != nmax - 1:
            lnpa_arr[nd, :] = lnpa_arr[nmax - 1, :].copy()
            get_psi(lnpa_arr, nd)

        lnpa_arr[nmax - 1, :] = np.array([-1.0, -1.0, -1.0, 0.0, -1.0, -1.0], dtype=float)

    # --- INITIALIZATION ---
    lnpa[:N_empty_LNP, 0] = R0 * (1.0 + 0.3 * np.random.normal(size=N_empty_LNP))
    lnpa[:N_empty_LNP, 1] = fiPEG0 * (1.0 + 0.3 * np.random.normal(size=N_empty_LNP))
    lnpa[:N_empty_LNP, 2] = 0.0
    lnpa[:N_empty_LNP, 4] = 0.0
    lnpa[:N_empty_LNP, 5] = 0.0

    start_idx = N_empty_LNP
    end_idx = N_empty_LNP + N_RNA_LNP
    if N_RNA_LNP > 0:
        lnpa[start_idx:end_idx, 0] = (R0**3 + R_RNA**3)**(1/3) * (1.0 + 0.3 * np.random.normal(size=N_RNA_LNP))
        lnpa[start_idx:end_idx, 1] = fiPEG0 * (1.0 + 0.3 * np.random.normal(size=N_RNA_LNP))
        lnpa[start_idx:end_idx, 4] = 1.0
        lnpa[start_idx:end_idx, 5] = 1.0
        for i in range(start_idx, end_idx):
            lnpa[i, 2] = lnpa[i, 5] * vRNA / (4.0/3.0 * math.pi * lnpa[i, 0]**3 * (1.0 - fw))

    start_idx = N_empty_LNP + N_RNA_LNP
    lnpa[start_idx:, 0] = R_RNA
    lnpa[start_idx:, 1] = 0.0
    lnpa[start_idx:, 2] = 1.0
    lnpa[start_idx:, 4] = 2.0
    lnpa[start_idx:, 5] = 1.0

    # Compute psi
    print("Computing initial surface potentials...")
    for i in range(N0):
        if lnpa[i, 0] > 0:
            get_psi(lnpa, i)
    
    # Check initial potentials
    psi_empty = np.mean(lnpa[lnpa[:, 4] == 0, 3])
    psi_rna = np.mean(lnpa[lnpa[:, 4] == 2, 3]) if N_free_RNA > 0 else 0
    print(f"Average psi (empty LNP): {psi_empty:.2f} mV")
    print(f"Average psi (free RNA): {psi_rna:.2f} mV")

    # Initialize rate matrix
    print("Initializing rate matrix...")
    rate = np.zeros((N0, N0), dtype=float)
    for i in range(N0):
        if lnpa[i, 0] > 0:
            get_rate1(lnpa, i, N0, rate[i, :])
        rate[i, i] = 0.0
    rate = 0.5 * (rate + rate.T)
    
    nonzero_rates = np.sum(rate > 0)
    total_possible = N0 * (N0 - 1) // 2
    print(f"Nonzero rates: {nonzero_rates}/{total_possible}")
    print(f"Max rate: {np.max(rate):.2e}")
    if nonzero_rates > 0:
        print(f"Min nonzero rate: {np.min(rate[rate > 0]):.2e}")

    choicesarr = [i for i in range(N0 * N0)]

    # --- KMC SIMULATION ---
    N = N0
    time = 0.0

    outfname = os.path.join(output_dir, 
                            f"timeseries_FRR{FRR}_cL{cL_ethanol}_try{try_id}_seed{seed}.dat")
    outfile = open(outfname, 'w')
    outfile.write(f"# FRR={FRR}, cL_ethanol={cL_ethanol}, epsilon_r={epsilon_r:.2f}, lB={lB:.4f}\n")
    outfile.write("# time N avR pd empty_frac rna_lnp_frac free_rna_frac\n")

    electrostatics_disabled = False
    wrote_18h_window = False
    outdwrite = False
    max_steps = 2000000
    step = 0
    
    print(f"Starting KMC simulation...")

    while N > 1 and time < 1e5 and step < max_steps:
        step += 1
        N2 = N * N
        
        ratesum = np.sum(rate[:N, :N]) / (2.0 * vol_system)
        if ratesum <= 0.0 or not np.isfinite(ratesum):
            print(f"No events possible (ratesum={ratesum:.2e}). Stopping at step {step}.")
            break

        tau = expovariate(ratesum)
        time += tau

        flat_rates = rate[:N, :N].flatten()
        if np.sum(flat_rates) <= 0.0:
            print("No available pair events. Exiting.")
            break

        act = choices(choicesarr[:N2], weights=flat_rates, k=1)[0]
        n1 = act % N
        n2 = act // N

        if n1 == n2:
            continue

        nm = n1 if n1 < n2 else n2
        nd = n1 if n1 > n2 else n2

        merge(lnpa, nm, nd, N)

        if nd != N - 1:
            rate[nd, :] = rate[N - 1, :].copy()
            rate[:, nd] = rate[:, N - 1].copy()
        rate[N - 1, :] = 0.0
        rate[:, N - 1] = 0.0
        rate[nd, nd] = 0.0

        N -= 1

        get_rate1(lnpa, nm, N, rate[nm, :N])
        rate[:N, nm] = rate[nm, :N]
        rate[nm, nm] = 0.0

        particle_types = lnpa[:N, 4].astype(int)
        n_empty = int(np.sum(particle_types == 0))
        n_rna_lnp = int(np.sum(particle_types == 1))
        n_free_rna = int(np.sum(particle_types == 2))
        n_lnp = n_empty + n_rna_lnp

        lnp_mask = particle_types != 2
        if np.sum(lnp_mask) > 0:
            avR = float(np.average(lnpa[:N, 0][lnp_mask]))
            pold = float(np.std(lnpa[:N, 0][lnp_mask]))
            pd = pold / avR if avR > 0 else 0.0
        else:
            avR = 0.0
            pd = 0.0

        empty_frac = n_empty / n_lnp if n_lnp > 0 else 0.0
        rna_lnp_frac = n_rna_lnp / n_lnp if n_lnp > 0 else 0.0
        free_rna_frac = n_free_rna / N if N > 0 else 0.0

        outfile.write(f"{time} {N} {avR} {pd} {empty_frac} {rna_lnp_frac} {free_rna_frac}\n")

        if 3600.0 * 16.0 < time < 3600.0 * 22.0 and not wrote_18h_window:
            snapshot_18h_fname = os.path.join(output_dir, 
                                              f"snapshot_18h_FRR{FRR}_cL{cL_ethanol}_try{try_id}_seed{seed}.dat")
            with open(snapshot_18h_fname, "w") as f:
                f.write(f"# FRR={FRR}, time={time:.2f}s\n")
                f.write("# idx r fiPEG fiRNA nRNA type\n")
                for j in range(N):
                    f.write(f"{j} {lnpa[j,0]:.4f} {lnpa[j,1]:.6f} {lnpa[j,2]:.6f} {lnpa[j,5]:.0f} {int(lnpa[j,4])}\n")
            wrote_18h_window = True
            print(f"18h snapshot at time={time:.2f}s")

        if time > 3600.0 and not electrostatics_disabled:
            gELON = 0
            for j in range(N):
                if lnpa[j, 0] > 0:
                    get_psi(lnpa, j)
            for i in range(N):
                get_rate1(lnpa, i, N, rate[i, :N])
                rate[i, i] = 0.0
            rate = 0.5 * (rate + rate.T)
            electrostatics_disabled = True
            print(f"Electrostatics OFF at time={time:.2f}s")

        if time > 3600.0 * 20.0 and not outdwrite:
            snapshot_20h_fname = os.path.join(output_dir, 
                                              f"snapshot_20h_FRR{FRR}_cL{cL_ethanol}_try{try_id}_seed{seed}.dat")
            with open(snapshot_20h_fname, "w") as f:
                f.write(f"# FRR={FRR}, time={time:.2f}s\n")
                f.write("# idx r fiPEG fiRNA nRNA type\n")
                for j in range(N):
                    f.write(f"{j} {lnpa[j,0]:.4f} {lnpa[j,1]:.6f} {lnpa[j,2]:.6f} {lnpa[j,5]:.0f} {int(lnpa[j,4])}\n")
            outdwrite = True
            print(f"20h snapshot at time={time:.2f}s")

        if step % 10000 == 0:
            print(f"Step {step}: t={time:.2f}s, N={N}, R={avR:.2f}nm")

    outfile.close()
    print(f"DONE: step={step}, time={time:.2f}s, N={N}\n")
    
    # Summary
    lnp_mask = lnpa[:N, 4] != 2
    if np.sum(lnp_mask) > 0:
        final_radii = lnpa[:N, 0][lnp_mask]
        avg_size = np.mean(final_radii[final_radii > 0])
        std_size = np.std(final_radii[final_radii > 0])
    else:
        avg_size = 0.0
        std_size = 0.0

    particle_types = lnpa[:N, 4].astype(int)
    n_empty_final = int(np.sum(particle_types == 0))
    n_rna_lnp_final = int(np.sum(particle_types == 1))
    n_free_rna_final = int(np.sum(particle_types == 2))

    return {
        "FRR": FRR,
        "water_frac": water_frac,
        "epsilon_r": epsilon_r,
        "lB": lB,
        "R0": R0,
        "cL_ethanol": cL_ethanol,
        "try_id": try_id,
        "seed": seed,
        "final_N": N,
        "final_time": time,
        "final_steps": step,
        "avg_size": avg_size,
        "std_size": std_size,
        "n_empty_LNP": n_empty_final,
        "n_rna_LNP": n_rna_lnp_final,
        "n_free_RNA": n_free_rna_final,
        "output_dir": output_dir,
        "timeseries_file": outfname
    }


if __name__ == "__main__":
    FRR_values = [3, 9]
    cL_ethanol_values = [10.0]
    R0_value = 10.0
    n_tries = 1
    
    results_summary = []
    
    for FRR in FRR_values:
        for cL_ethanol in cL_ethanol_values:
            for try_id in range(n_tries):
                seed = try_id * 1000 + FRR * 100 + int(cL_ethanol * 10)
                
                result = run_simulation(
                    FRR=FRR,
                    cL_ethanol=cL_ethanol,
                    R0=R0_value,
                    N1=2000,
                    try_id=try_id,
                    seed=seed
                )
                
                results_summary.append(result)
    
    summary_file = "simulation_summary_FRR.dat"
    with open(summary_file, 'w') as f:
        f.write("# FRR water% epsilon lB cL_eth try seed N time size std empty rna free\n")
        for res in results_summary:
            f.write(f"{res['FRR']} {res['water_frac']*100:.1f} {res['epsilon_r']:.2f} {res['lB']:.4f} ")
            f.write(f"{res['cL_ethanol']:.1f} {res['try_id']} {res['seed']} ")
            f.write(f"{res['final_N']} {res['final_time']:.2f} {res['avg_size']:.2f} {res['std_size']:.2f} ")
            f.write(f"{res['n_empty_LNP']} {res['n_rna_LNP']} {res['n_free_RNA']}\n")
    
    print(f"\nSummary saved to {summary_file}")
import numpy as np
import math
from random import choices, expovariate
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import os
import joblib

try:
    model_data = joblib.load("psi_neural_network.pkl")
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    USE_NN = True
    print("Loaded neural network model")
except:
    USE_NN = False
    print(" No NN model - using analytical formula")

R_values = np.linspace(1, 30, 50)
XRNA_values = np.linspace(0, 1, 50)


def get_psi_interpolator(csalt_val):

    R_grid, XRNA_grid = np.meshgrid(R_values, XRNA_values, indexing="ij")
    input_data = np.column_stack([
        R_grid.ravel(),
        XRNA_grid.ravel(),
        np.full(R_grid.size, csalt_val)
    ])
    
    input_scaled = scaler_X.transform(input_data)
    psi_pred_scaled = model.predict(input_scaled)
    psi_pred = scaler_y.inverse_transform(psi_pred_scaled.reshape(-1, 1)).ravel()
    psi_pred = psi_pred.reshape(R_grid.shape)
    
    return RegularGridInterpolator((R_values, XRNA_values), psi_pred)

def run_simulation_pei_sirna(csalt,N0=2500, try_id=1, seed=None,NP_ratio_target=6.0, sirna_conc_ug_per_mL=1.0):
    if seed is not None:
        np.random.seed(seed)

    # ---- constants ----
    eta = 0.001
    Aham = 0.7
    temperature = 298
    lB = 0.71
    gELON = 1
    fw=0.2

    # Debye length
    lD = 1 / np.sqrt(8 * math.pi * lB * 0.60223 * csalt)

    psi_interp_func = get_psi_interpolator(csalt)

    # ---- siRNA / PEI parameters; (change values for other cargos) ----
    siRNA_bp = 2200 # this is 4.4 kb DNA, not sirna
    siRNA_nt = 2 * siRNA_bp + 4
    phosphates_per_siRNA = siRNA_nt

    siRNA_length = siRNA_bp * 0.3
    siRNA_diameter = 2.3
    V_siRNA = math.pi * (siRNA_diameter/2)**2 * siRNA_length
    R_siRNA = (3 * V_siRNA / (4 * math.pi))**(1/3)

    PEI_MW = 25000
    MW_monomer = 43
    nitrogens_per_PEI = PEI_MW / MW_monomer
    R_PEI = 0.30 * nitrogens_per_PEI ** 0.588
    vRNA = V_siRNA
    vPEI = (4.0/3.0) * math.pi * R_PEI**3

    PEI_per_siRNA = NP_ratio_target * phosphates_per_siRNA / nitrogens_per_PEI

    # particle counts at t=0
    N_siRNA = int(round(N0 / (1.0 + PEI_per_siRNA)))
    N_PEI = N0 - N_siRNA

    # ---- system volume from siRNA concentration (1 ug/uL) ----
    #MW ~ 300.3 Da for 1nt (rough).
    MW_siRNA_g_per_mol = 300*siRNA_bp*2

    NA = 6.02214076e23
    sirna_g_per_L = sirna_conc_ug_per_mL * 1e-3   
    sirna_mol_per_L = sirna_g_per_L / MW_siRNA_g_per_mol
    sirna_per_L = sirna_mol_per_L * NA
    vol_system_L = N_siRNA / sirna_per_L
    vol_system = vol_system_L 

    # ---- lnpa: [R, fipeg, firna, psi] ----
    lnpa = np.full((N0, 4), -1.0)

    phosphates_per_siRNA = siRNA_nt


    PEI_per_siRNA = NP_ratio_target * phosphates_per_siRNA / nitrogens_per_PEI

    N_rna = int(round(N0 / (1.0 + PEI_per_siRNA)))
    # 1) set radii
    lnpa[:N_rna, 0] = R_siRNA
    lnpa[N_rna:, 0] = R_PEI

    lnpa[:, 1] = 0 #no peg
    lnpa[:N_rna, 2] = 1 #all rna
    lnpa[N_rna:, 2] = 0 #all pei
    lnpa[:, 3] = 0.0



    def get_psi(lnpa, i):
        R = np.clip(lnpa[i, 0], R_values[0], R_values[-1])
        xrna = np.clip(lnpa[i, 2], XRNA_values[0], XRNA_values[-1])
        psi = psi_interp_func(np.array([R, xrna]))
        lnpa[i, 3] = psi * gELON
        return lnpa[i, 3]

    def mW(D, n1, n2):
        W = -Aham / 6 * lnpa[n1, 0] * lnpa[n2, 0] / (lnpa[n1, 0] + lnpa[n2, 0]) / D + \
            80 * 8.85e-4 / (1.38 * temperature * 4) * lnpa[n1, 0] * lnpa[n2, 0] / (lnpa[n1, 0] + lnpa[n2, 0]) * \
            (lnpa[n1, 3] ** 2 + lnpa[n2, 3] ** 2) * \
            (2 * lnpa[n1, 3] * lnpa[n2, 3] / (lnpa[n1, 3] ** 2 + lnpa[n2, 3] ** 2) *
             np.log((1 + np.exp(-D / lD)) / (1 - np.exp(-D / lD))) + np.log(1 - np.exp(-2 * D / lD)))
        return -W

    def get_rate1(n1, nmax, rate):
        Wbmax = np.zeros(nmax)
        if abs(lnpa[n1, 3]) > 5:
            for j in range(nmax):
                if lnpa[n1, 3] * lnpa[j, 3] > 200:
                    Wbmax[j] = -mW(np.linspace(0.1, 2, num=50), n1, j).min()
        cfactor = 1.38e-23 * temperature 
        rate[:nmax] = cfactor * 2 / (3 * eta) * (lnpa[n1, 0] + lnpa[:nmax, 0]) ** 2 / (lnpa[n1, 0] * lnpa[:nmax, 0]) * np.exp( - Wbmax)
            
        tol = 1e-12
        x1 = lnpa[n1, 2]
        x2 = lnpa[:nmax, 2]

        # PEI-only <-> RNA-containing (RNA-only or complex)
        allowed = ((x1 < tol) & (x2 > tol)) | ((x1 > tol) & (x2 < tol))

        # RNA-only <-> complex (RNA binds complex)
        allowed |= ((x1 > 1 - tol) & (x2 > tol) & (x2 < 1 - tol))
        allowed |= ((x1 > tol) & (x1 < 1 - tol) & (x2 > 1 - tol))
        allowed |= ((x1 > tol) & (x1 < 1 - tol) & (x2 > tol) & (x2 < 1 - tol))

        rate[:nmax] *= allowed
        rate[n1] = 0.0


    def merge(n1, n2, nmax):
        rnew = (lnpa[n1, 0] ** 3 + lnpa[n2, 0] ** 3) ** (1 / 3)
        fiPEGnew = 0
        fiRNAnew = (lnpa[n1, 2] * lnpa[n1, 0] ** 3 + lnpa[n2, 2] * lnpa[n2, 0] ** 3) / (lnpa[n1, 0] ** 3 + lnpa[n2, 0] ** 3)
        keep, kill = sorted([n1, n2])
        lnpa[keep] = np.array([rnew, fiPEGnew, fiRNAnew, 0])
        lnpa[kill] = lnpa[nmax - 1]
        lnpa[nmax - 1] = -1
        get_psi(lnpa, keep)
        return keep                  

    # initialize psi
    for i in range(N0):
        get_psi(lnpa,i)

    rate = np.zeros((N0, N0))
    for i in range(N0):
        get_rate1(i, N0, rate[i, :])
        rate[i, i] = 0

    choicesarr = np.arange(N0 * N0)
    time, N = 0.0, N0
    time_data, size_data, empty_data, var_data, mean_data, count_data = [], [], [], [], [], []

    out_dir = "results_400"
    os.makedirs(out_dir, exist_ok=True)
    outfile = open(os.path.join(out_dir, f"try{try_id}_NP{NP_ratio_target}_EL{gELON}.dat"), "w")
    wrote_1s_window = False

    while N > 1 and time < 1e10:
        ratesum = np.sum(rate[:N, :N]) / (2 * vol_system)
        if ratesum == 0:
            break
        tau = expovariate(ratesum)
        act = choices(choicesarr[:N * N], weights=rate[:N, :N].flatten(), k=1)[0]
        n1, n2 = act % N, act // N
        if n1 == n2: continue

        keep = merge(n1, n2, N)          # <-- capture keep
        psi_new = lnpa[keep, 3]          # psi of newly created particle

        rate[n2, :] = rate[N - 1, :]
        rate[:, n2] = rate[:, N - 1]
        rate[N - 1, :] = rate[:, N - 1] = 0
        N -= 1
        time += tau

        # update n1 rates
        get_rate1(n1, N, rate[n1, :])
        rate[:N, n1] = rate[n1, :N]
        rate[n1, n1] = 0

        if n2 < N:
            get_rate1(n2, N, rate[n2, :])
            rate[:N, n2] = rate[n2, :N]
            rate[n2, n2] = 0

        avR = np.average(lnpa[:N, 0])
        pdd = np.std(lnpa[:N, 0]) / avR
        firna = lnpa[:N, 2]
        rna = lnpa[:N, 2] * 4 * math.pi / 3 * lnpa[:N, 0] ** 3 / vRNA * (1 - fw)
        empty = np.count_nonzero(firna == 0) / N

        time_data.append(time)
        size_data.append(avR)
        empty_data.append(empty)
        non_empty_rna = rna[rna > 0]
        count_data.append(len(non_empty_rna))
        mean_data.append(np.mean(non_empty_rna))
        var_data.append(np.var(non_empty_rna))
        firna = lnpa[:N, 2]

        nRNA = int(np.sum(firna == 1.0))                 # free siRNA particles
        nPEI = int(np.sum(firna == 0.0))                 # free PEI particles
        nPEC = int(np.sum((firna > 0.0) & (firna < 1.0)))# complexes
        psi_new = lnpa[keep, 3]

        outfile.write(f"{time} {avR} {pdd} {empty} {nRNA} {nPEI} {nPEC} {psi_new}\n")

        V_PEI = (4.0 * math.pi / 3.0) * (R_PEI ** 3)
        V_RNA = (4.0 * math.pi / 3.0) * (R_siRNA ** 3)

        if (0.95 < time < 1.05) and (not wrote_1s_window):

            snapshot_1s_fname = os.path.join(
                out_dir,
                f"snapshot_1s_NP{NP_ratio_target}_try{try_id}_seed{seed}.dat"
            )

            with open(snapshot_1s_fname, "w") as f:
                f.write("# idx r_nm xrna nrna npei psi\n")
                for j in range(N):
                    r = float(lnpa[j, 0])      # complex radius
                    xrna = float(lnpa[j, 2])   # RNA volume fraction
                    psi = float(lnpa[j, 3])    # psi

                    V_complex = (4.0 * math.pi / 3.0) * (r ** 3)

                    # counts by volume partitioning
                    nrna = V_complex * xrna / V_RNA
                    npei = V_complex * (1.0 - xrna) / V_PEI

                    f.write(f"{j} {r:.6f} {xrna:.8f} {nrna:.6f} {npei:.6f} {psi:.8f}\n")
            print(f"1s snapshot written at time={time:.3f}s -> {snapshot_1s_fname}")
    outfile.close()


if __name__ == "__main__":
    # Example run parameters
    c_salt = 0.15  # e.g., 150 mM

    run_simulation_pei_sirna(
        csalt=.1,
        N0=2000,
        try_id=21,
        seed=123,
        NP_ratio_target=6.0,
        sirna_conc_ug_per_mL=100.0
    )

    print("Done. Output written to results/")
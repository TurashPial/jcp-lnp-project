# JCP LNP Project

Mechanistic Framework for Multicomponent Nanoparticle Assembly: Predicting RNA-lipid and PEI-DNA nanoparticle assembly

## Project contents

```text
code/
  pbe_grid_simulation.py          Population balance simulation for LNP size evolution with CNT andd growth
  lnp_growth_no_rna.py            LNP growth model without RNA; before mixing time
  mixing_length_analysis.py       Mixing length calculation for LNP and mRNA diffusion
  kolmogorov_lengthscale.py       Flow-rate dependent length scale calculation
  LNP_KMC.py                      Kinetic Monte Carlo simulation
  run_parallel_kmc.py             Parallel runner for KMC simulations
  train_surface_potential_nn.py   Neural network model for nanoparticle surface potential with charge regulation
  pei_DNA_polyplex_kmc.py       KMC simulation for PEI-DNA polyplex formation. 

data/
  psi_vs_R_25mM_noRNA.csv
  mixing_target_summary_5mM_10mM.csv
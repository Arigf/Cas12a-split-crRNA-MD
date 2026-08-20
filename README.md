# Cas12a-split-crRNA-MD
Molecular dynamics trajectories and analysis scripts for Cas12a split-crRNA systems.

MD Analysis Script : analysis_202608.

MD results : https://zenodo.org/records/22024385?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2Y2Y1NWUxLTMyY2QtNDc3Zi1iZjQwLTE4OTFlZDBiNDg3NCIsImRhdGEiOnt9LCJyYW5kb20iOiI1MWRmYTljOTlkMzU4Y2YwNTI0YmM1MDBmMjVmNzU0ZiJ9.dj7YZu6WLKfE7GNi3EYPJDFFqFZ8P4IoAeszXbxwBSGmtVOVa6j_HGNuW0oXxbCSqJ_ipExeB0SJBmqVY4eGqQ.



## MD Method
### Step 1. AlphaFold3 prediction

LbCas12a complexes were predicted using AlphaFold3 v.3.0.3  with LbCas12a protein, target DNA, and crRNA as inputs. Four systems were modeled: FL-PM, Split-PM, FL-MM17, and Split-MM17. The highest-confidence structure of each system was selected for MD simulations.

### Step 2. MD system preparation

MD simulations were performed using OpenMM 8.5.2 , the Amber14 force field , and TIP3P water . Split-crRNA termini were modeled as 5′-OH ends, and two catalytic Mg²⁺ ions were positioned in the RuvC domain based on PDB 8SFO. Each system was solvated in a cubic box with 1.2 nm padding and supplemented with 2.5 mM MgCl₂, 37.5 mM KCl, and additional K⁺ for charge neutralization.

### Step 3. Minimization, equilibration, and production MD

Each system underwent staged energy minimization, followed by 20 ps NVT heating to 310.15 K, 50 ps restrained NPT equilibration, and 20 ps unrestrained NPT equilibration. Production MD was performed for 100 ns at 310.15 K and 1 atm with a 2 fs timestep.

### Step 4. Trajectory analysis

MD trajectories were analyzed using MDTraj, MDAnalysis, NumPy, pandas, and custom scripts. Base-pairing proxies, RuvC-domain distances, and RuvC-entry metrics were calculated to compare the four systems.

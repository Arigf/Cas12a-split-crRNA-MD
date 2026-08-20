# Cas12a-split-crRNA-MD
Molecular dynamics trajectories and analysis scripts for Cas12a split-crRNA systems.

MD Analysis Script : analysis_202608.

MD results : https://zenodo.org/records/22024385?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2Y2Y1NWUxLTMyY2QtNDc3Zi1iZjQwLTE4OTFlZDBiNDg3NCIsImRhdGEiOnt9LCJyYW5kb20iOiI1MWRmYTljOTlkMzU4Y2YwNTI0YmM1MDBmMjVmNzU0ZiJ9.dj7YZu6WLKfE7GNi3EYPJDFFqFZ8P4IoAeszXbxwBSGmtVOVa6j_HGNuW0oXxbCSqJ_ipExeB0SJBmqVY4eGqQ.


### Naming Abbreviation
OsDEP1-MM17-FL : MM-Full.

OsDEP1-MM17-Split : MM-Split.

OsDEP1-PM-FL : Match-Full.

OsDEP1-PM-Split : Match-Split.


## MD Method
### Step 1. AlphaFold3 prediction

LbCas12a complexes were predicted using AlphaFold3 v.3.0.3 (ref 1) with LbCas12a protein, target DNA, and crRNA as inputs. Four systems were modeled: FL-PM, Split-PM, FL-MM17, and Split-MM17. The highest-confidence structure of each system was selected for MD simulations.

### Step 2. MD system preparation

MD simulations were performed using OpenMM 8.5.2 (ref 2), the Amber14 force field (refs 3–5), and TIP3P water (ref 6). Split-crRNA termini were modeled as 5′-OH ends, and two catalytic Mg²⁺ ions were positioned in the RuvC domain based on PDB 8SFO (ref 7). Each system was solvated in a cubic box with 1.2 nm padding and supplemented with 2.5 mM MgCl₂, 37.5 mM KCl, and additional K⁺ for charge neutralization.

### Step 3. Minimization, equilibration, and production MD

Each system underwent staged energy minimization, followed by 20 ps NVT heating to 310.15 K, 50 ps restrained NPT equilibration, and 20 ps unrestrained NPT equilibration. Production MD was performed for 100 ns at 310.15 K and 1 atm with a 2 fs timestep.

### Step 4. Trajectory analysis

MD trajectories were analyzed using MDTraj, MDAnalysis, NumPy, pandas, and custom scripts. Base-pairing proxies, RuvC-domain distances, and RuvC-entry metrics were calculated to compare the four systems.
## Reference

Ref1
Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 2024, 630 (8016), 493–500. DOI: 10.1038/s41586-024-07487-w

Ref2
OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials. J. Phys. Chem. B 2023, 128 (1), 109–116

Ref3
ff14SB: Improving the accuracy of protein side chain and backbone parameters from ff99SB. J. Chem. Theory Comput. 2015, 11 (8), 3696–3713. DOI: 10.1021/acs.jctc.5b00255

Ref4
Refinement of the sugar-phosphate backbone torsion beta for AMBER force fields improves the description of Z- and B-DNA. J. Chem. Theory Comput. 2015, 11 (12), 5723–5736. DOI: 10.1021/acs.jctc.5b00706

Ref5
Refinement of the Cornell et al. nucleic acids force field based on reference quantum chemical calculations of glycosidic torsion profiles. J. Chem. Theory Comput. 2011, 7 (9), 2886–2902. DOI: 10.1021/ct200162x

Ref6
Comparison of simple potential functions for simulating liquid water. J. Chem. Phys. 1983, 79 (2), 926–935.DOI: 10.1063/1.445869

Ref7
Strohkendl, I., Saha, A., Moy, C., Nguyen, A. H., Ahsan, M., Russell, R., Palermo, G., & Taylor, D. W. (2024). Cas12a domain flexibility guides R-loop formation and forces RuvC resetting. Molecular Cell, 84(14), 2717-2731.e6. https://doi.org/10.1016/j.molcel.2024.06.007

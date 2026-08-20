# System Description

This dataset contains the first-batch Cas12a MD systems used to compare full-length versus split crRNA and perfect-match versus M17-mismatch guide-target hybrids.

## Systems

| Dataset name | Short label | Description |
|---|---|
| `OsDEP1-PM-FL` |  Full-length crRNA with perfect-match guide-target pairing. |
| `OsDEP1-PM-Split` |  Split crRNA with perfect-match guide-target pairing. |
| `OsDEP1-MM17-FL` |  Full-length crRNA with the M17 mismatch. |
| `OsDEP1-MM17-Split` |  Split crRNA with the M17 mismatch. |

## Chain and Register Definitions

- Chain `A`: Cas12a protein.
- Chain `B`: target DNA strand.
- Full-crRNA systems: crRNA is chain `C`, residues 1-44.
- Split-crRNA systems: front crRNA segment is chain `C`, residues 1-34; back crRNA segment is chain `D`, residues 1-10.
- Guide-equivalent positions are `eq1-eq44`.
- In split-crRNA systems, chain `D` residues 1-10 correspond to guide-equivalent positions `eq35-eq44`.
- Target mapping used in the analysis: target DNA chain `B` residue = `64 - guide_equivalent_position`.
- Under this mapping, `eq14-eq44` map to target residues `B50-B20`; `eq1-eq13` have no mapped target residue in chain `B`.

## MD Setup

The first-batch simulations used OpenMM with:

- `amber14-all.xml`
- `amber14/tip3p.xml`
- TIP3P water
- PME electrostatics
- 1.0 nm nonbonded cutoff
- HBonds constraints and rigid water
- 310.15 K
- 1 atm
- 2 fs timestep
- nominal 2.5 mM MgCl2 and 37.5 mM KCl

The production trajectories are solute-only DCD files. Each DCD should be read together with the matching `*_solute_atom_indices.txt` file and the corresponding production final PDB.


#!/usr/bin/env python3
"""Extract aligned MM-Split eq39 dissociation and Match-Full control frames."""

from pathlib import Path
import sys
import mdtraj as md
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ruvc_entry_grid as ag
import analyze_precatalytic_allostery as pa

OUT=ROOT/'analysis_four_systems_20_100ns/eq39_keyframes_comparison'
FEATURES=ROOT/'analysis_four_systems_20_100ns/precatalytic_allostery/feature_timeseries.csv'
TIMES=[26.0,27.0,65.0,97.0]
LABELS={26.0:'paired_before_transition',27.0:'first_sustained_dissociation',
        65.0:'stable_dissociated_representative',97.0:'maximum_dissociation'}
CONTROL_LABELS={26.0:'time_matched_pretransition_control',27.0:'time_matched_transition_control',
                65.0:'time_matched_stable_control',97.0:'time_matched_maximum_control'}


def local_indices(top,system):
    guide_chain='D' if system=='MM-Split' else 'C'
    guide_res=range(3,7) if system=='MM-Split' else range(37,41)
    return np.array([a.index for a in top.atoms
                     if (a.residue.chain.chain_id=='B' and 24<=a.residue.resSeq<=27)
                     or (a.residue.chain.chain_id==guide_chain and a.residue.resSeq in guide_res)],int)


def write_pml(path):
    lines=[
      'reinitialize',
      'set dash_width, 2.0','set dash_gap, 0.25','set stick_radius, 0.16',
      'bg_color white','set ray_opaque_background, off',
      'load local/MM-Split_26ns_paired_before_transition_local.pdb, MMS_26',
      'load local/MM-Split_27ns_first_sustained_dissociation_local.pdb, MMS_27',
      'load local/MM-Split_65ns_stable_dissociated_representative_local.pdb, MMS_65',
      'load local/MM-Split_97ns_maximum_dissociation_local.pdb, MMS_97',
      'load local/Match-Full_26ns_time_matched_pretransition_control_local.pdb, MF_26',
      'load local/Match-Full_27ns_time_matched_transition_control_local.pdb, MF_27',
      'load local/Match-Full_65ns_time_matched_stable_control_local.pdb, MF_65',
      'load local/Match-Full_97ns_time_matched_maximum_control_local.pdb, MF_97',
      'hide everything','show sticks, all','color gray70, chain B','color marine, MF_* and chain C',
      'color magenta, MMS_* and chain D','color tv_red, chain B and resi 25',
      'color violet, (MF_* and chain C and resi 39) or (MMS_* and chain D and resi 5)',
      'group MM_Split_transition, MMS_26 MMS_27 MMS_65 MMS_97',
      'group Match_Full_controls, MF_26 MF_27 MF_65 MF_97',
      'disable all','enable MMS_26','enable MMS_27','enable MF_26','enable MF_27',
      'orient','zoom all, 4','set orthoscopic, on',
      '# Objects are already aligned to the common Match-Full protein reference frame.',
      '# Toggle the grouped objects to compare the stable (65 ns) and maximal (97 ns) states.',
    ]
    path.write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'full_complex').mkdir(exist_ok=True);(OUT/'local').mkdir(exist_ok=True)
    feat=pd.read_csv(FEATURES);ref=pa.global_reference();rows=[]
    for system in ['MM-Split','Match-Full']:
        tr,top,time=ag.load_system(system);xyz=pa.align_global(tr,top,ref);local=local_indices(top,system)
        selected=[];selected_local=[]
        for t in TIMES:
            fi=int(np.where(np.isclose(time,t))[0][0]);label=(LABELS[t] if system=='MM-Split' else CONTROL_LABELS[t])
            one=md.Trajectory(xyz[fi:fi+1]/10.0,top);selected.append(one)
            loc=one.atom_slice(local);selected_local.append(loc)
            stem=f'{system}_{int(t)}ns_{label}'
            one.save_pdb(str(OUT/'full_complex'/f'{stem}_full_aligned.pdb'))
            loc.save_pdb(str(OUT/'local'/f'{stem}_local.pdb'))
            r=feat[(feat.system==system)&np.isclose(feat.time_ns,t)].iloc[0]
            rows.append({'system':system,'time_ns':t,'role':label,
                         'eq39_B25_basecentroid_A':r.hybrid_eq39_basecentroid_A,
                         'eq39_B25_min_base_A':r.hybrid_eq39_min_base_A,
                         'eq39_B25_NO_contacts_lt3p5A':int(r.hybrid_eq39_NO_contacts_3p5A),
                         'paired_state':bool((r.hybrid_eq39_basecentroid_A<6.5) and (r.hybrid_eq39_NO_contacts_3p5A>=1)),
                         'full_pdb':f'full_complex/{stem}_full_aligned.pdb','local_pdb':f'local/{stem}_local.pdb'})
        md.join(selected,check_topology=True,discard_overlapping_frames=False).save_pdb(str(OUT/f'{system}_four_keyframes_full_aligned.pdb'))
        md.join(selected_local,check_topology=True,discard_overlapping_frames=False).save_pdb(str(OUT/f'{system}_four_keyframes_local.pdb'))
    pd.DataFrame(rows).to_csv(OUT/'frame_selection_and_metrics.csv',index=False)
    write_pml(OUT/'compare_eq39_B25.pml')
    readme='''# eq39–B25 key frames

All structures were imaged and aligned to the common Match-Full protein C-alpha
reference. MM-Split and Match-Full were extracted at identical times.

- 26 ns: final paired MM-Split frame before the persistent transition.
- 27 ns: first frame of sustained MM-Split dissociation.
- 65 ns: frame closest to the median post-transition MM-Split distance.
- 97 ns: maximum MM-Split base-centroid separation.

`full_complex/` contains complete solute complexes. `local/` contains target-DNA
B24–B27 plus equivalent guide positions 37–40 (MM-Split D3–D6; Match-Full
C37–C40). The two multi-model PDB files contain the four frames in chronological
order. Open `compare_eq39_B25.pml` from this directory in PyMOL for a prepared
comparison. Exact selection metrics and file names are in
`frame_selection_and_metrics.csv`.
'''
    (OUT/'README.md').write_text(readme)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()

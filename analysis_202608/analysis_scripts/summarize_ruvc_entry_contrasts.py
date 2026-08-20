#!/usr/bin/env python3
"""Create descriptive 2x2 contrasts for the RuvC entry analysis."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_four_systems_20_100ns/ruvc_entry_grid_analysis'


def main():
    d=pd.read_csv(OUT/'timeseries_20_100ns_1ns.csv')
    metrics=[
      'protein_NTS_primary_min_clearance_A',
      'protein_NTS_primary_accessible_volume_probe1.4_A3',
      'protein_NTS_primary_accessible_volume_probe2.5_A3',
      'protein_TS_sensitivity_min_clearance_A',
      'protein_TS_sensitivity_accessible_volume_probe1.4_A3',
      'protein_TS_sensitivity_accessible_volume_probe2.5_A3']
    contrasts={
      'split_effect_in_match':('Match-Split','Match-Full'),
      'split_effect_in_mismatch':('MM-Split','MM-Full'),
      'mismatch_effect_in_full':('MM-Full','Match-Full'),
      'mismatch_effect_in_split':('MM-Split','Match-Split')}
    rows=[]
    for metric in metrics:
      blocks={s:np.asarray([x.mean() for x in np.array_split(d[d.system==s][metric].to_numpy(),8)]) for s in d.system.unique()}
      for name,(a,b) in contrasts.items():
        delta=blocks[a]-blocks[b]
        rows.append({'metric':metric,'contrast':name,'system_a':a,'system_b':b,
                     'mean_delta_a_minus_b':d[d.system==a][metric].mean()-d[d.system==b][metric].mean(),
                     'block_delta_p2p5':np.percentile(delta,2.5),'block_delta_p97p5':np.percentile(delta,97.5)})
      interaction=(blocks['MM-Split']-blocks['MM-Full'])-(blocks['Match-Split']-blocks['Match-Full'])
      rows.append({'metric':metric,'contrast':'split_x_mismatch_interaction','system_a':'(MM-Split-MM-Full)',
                   'system_b':'(Match-Split-Match-Full)','mean_delta_a_minus_b':interaction.mean(),
                   'block_delta_p2p5':np.percentile(interaction,2.5),'block_delta_p97p5':np.percentile(interaction,97.5)})
    pd.DataFrame(rows).to_csv(OUT/'two_by_two_contrasts.csv',index=False)

    owners=[]
    for col in [x for x in d.columns if x.endswith('bottleneck_owner')]:
      for system in d.system.unique():
        res=d.loc[d.system==system,col].str.rsplit(':',n=1).str[0]
        for owner,count in res.value_counts().items():
          owners.append({'system':system,'metric':col,'owner_residue':owner,'frames':count,'fraction':count/len(res)})
    pd.DataFrame(owners).to_csv(OUT/'bottleneck_owner_residue_frequencies.csv',index=False)
    print('wrote contrasts and residue-level owner frequencies')


if __name__=='__main__': main()

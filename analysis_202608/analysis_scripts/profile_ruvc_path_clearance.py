#!/usr/bin/env python3
"""Generate position-resolved RuvC path-clearance profiles without rerunning grids."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_ruvc_entry_grid as ag


def arclength(path):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def main():
    out = ag.OUT
    paths = {
        "NTS_primary": np.loadtxt(ag.REFDIR / "ruvc_primary_nts_path.csv", delimiter=",", skiprows=1),
        "TS_sensitivity": np.loadtxt(ag.REFDIR / "ruvc_sensitivity_ts_path.csv", delimiter=",", skiprows=1),
    }
    ref = ag.reference_ca()
    rows=[]
    profiles={}
    for system in ag.SYSTEMS:
        tr, top, times = ag.load_system(system)
        xyz, _ = ag.align_frames(tr, top, ref)
        ids = ag.wall_indices(top, "protein")
        atomlist = list(top.atoms)
        radii = np.asarray([ag.VDW.get(ag.element(atomlist[i]),1.70) for i in ids])
        for pname,path in paths.items():
            curves=[]
            for f,time in enumerate(times):
                clear,_ = ag.path_clearance(path, xyz[f,ids], radii)
                curves.append(clear)
                for i,v in enumerate(clear):
                    rows.append({"system":system,"time_ns":time,"path":pname,
                                 "point_index":i,"arclength_A":arclength(path)[i],"clearance_A":v})
            profiles[(system,pname)] = np.asarray(curves)
    raw=pd.DataFrame(rows);raw.to_csv(out/'path_clearance_profiles_raw.csv',index=False)
    agg=(raw.groupby(['system','path','point_index','arclength_A']).clearance_A
         .agg(mean_A='mean',sd_A='std',p05_A=lambda x:np.percentile(x,5),
              median_A='median',p95_A=lambda x:np.percentile(x,95),
              open_fraction_1p4=lambda x:np.mean(x>=1.4),
              open_fraction_2p5=lambda x:np.mean(x>=2.5)).reset_index())
    agg.to_csv(out/'path_clearance_profiles_summary.csv',index=False)

    colors={"Match-Full":"#1f77b4","Match-Split":"#2ca02c","MM-Full":"#ff7f0e","MM-Split":"#d62728"}
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharey=True)
    for ax,(pname,path) in zip(axes,paths.items()):
        s=arclength(path)
        for system in ag.SYSTEMS:
            x=profiles[(system,pname)];mean=x.mean(0);lo=np.percentile(x,5,axis=0);hi=np.percentile(x,95,axis=0)
            ax.plot(s,mean,label=system,color=colors[system],lw=1.8)
            ax.fill_between(s,lo,hi,color=colors[system],alpha=.12)
        ax.axhline(1.4,color='k',ls='--',lw=.8,label='1.4 A probe' if pname=='NTS_primary' else None)
        ax.axhline(2.5,color='gray',ls=':',lw=.8,label='2.5 A probe' if pname=='NTS_primary' else None)
        ax.set_title(pname);ax.set_ylabel('surface clearance (A)');ax.grid(alpha=.2)
    axes[-1].set_xlabel('path arclength (A)');axes[0].legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(out/'path_clearance_profiles.png',dpi=200);plt.close(fig)
    print(json.dumps({"raw_rows":len(raw),"summary_rows":len(agg)},indent=2))


if __name__=='__main__':
    main()

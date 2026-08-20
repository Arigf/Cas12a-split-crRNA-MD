#!/usr/bin/env python3
"""Internal convergence diagnostics for the four single-trajectory MD systems."""

from pathlib import Path
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'analysis_four_systems_20_100ns/precatalytic_allostery'
ENT=ROOT/'analysis_four_systems_20_100ns/ruvc_entry_grid_analysis'
PROD=ROOT/'production_md_2p5mM_MgCl2_37p5mM_KCl'
OUT=ROOT/'analysis_four_systems_20_100ns/convergence_analysis'
SYSTEMS=['Match-Full','Match-Split','MM-Full','MM-Split']
COLORS={'Match-Full':'#0072B2','Match-Split':'#009E73','MM-Full':'#E69F00','MM-Split':'#D55E00'}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})


def acf_tau(x):
    x=np.asarray(x,float);x=x-x.mean();n=len(x)
    if n<3 or np.allclose(x,0): return np.ones(n),1.0,float(n)
    ac=np.correlate(x,x,mode='full')[n-1:]/np.arange(n,0,-1);ac=ac/ac[0]
    positive=[]
    for v in ac[1:]:
        if v<=0:break
        positive.append(v)
    tau=max(1.0,1+2*np.sum(positive));return ac,tau,n/tau


def diagnostics(system,metric,time,x,start=None):
    time=np.asarray(time);x=np.asarray(x,float)
    if start is not None: keep=time>=start;time=time[keep];x=x[keep]
    ac,tau,ess=acf_tau(x);half=len(x)//2;a=x[:half];b=x[half:]
    last=min(30,len(x));slope=np.polyfit(time[-last:],x[-last:],1)[0]
    sd=x.std();return {'system':system,'metric':metric,'start_ns':float(time[0]),'end_ns':float(time[-1]),
      'n_samples':len(x),'mean':x.mean(),'sd':sd,'tau_samples':tau,'effective_sample_size':ess,
      'first_half_mean':a.mean(),'second_half_mean':b.mean(),'half_delta':b.mean()-a.mean(),
      'half_delta_in_SD':(b.mean()-a.mean())/sd if sd else 0,
      'last30_sample_slope_per_ns':slope,
      'last30_total_change_in_SD':slope*(time[-1]-time[-last])/sd if sd else 0}


def block_series(time,x,width=10,start=20):
    rows=[]
    for lo in np.arange(start,100,width):
        m=(time>=lo)&(time<lo+width)
        if m.any():rows.append((lo+width/2,float(np.mean(x[m]))))
    return rows


def panel(ax,label): ax.text(.015,.97,label,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top',zorder=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.5))
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=3)


def save(fig,name):
    for ext,dpi in [('png',600),('pdf',300),('svg',300)]:fig.savefig(OUT/f'{name}.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pre=pd.read_csv(PRE/'feature_timeseries.csv');entry=pd.read_csv(ENT/'timeseries_20_100ns_1ns.csv')
    brief={s:pd.read_csv(PROD/s/'analysis_brief/timeseries_20_100ns.csv') for s in SYSTEMS}
    diag=[]
    for s in SYSTEMS:
        p=pre[pre.system==s];e=entry[entry.system==s];b=brief[s]
        for metric,t,x,start in [
          ('protein_rmsd_A',b.time_ns,b.protein_rmsd_A,20),('pocket_rmsd_A',b.time_ns,b.pocket_rmsd_A,20),
          ('Mg_Mg_A',b.time_ns,b.Mg_Mg_A,20),('eq39_B25_basecentroid_A',p.time_ns,p.hybrid_eq39_basecentroid_A,30),
          ('RuvCI_RuvCIII_COM_A',p.time_ns,p.RuvCI_RuvCIII_COM_A,20),
          ('NTS_volume_probe1p4_A3',e.time_ns,e['protein_NTS_primary_accessible_volume_probe1.4_A3'],20)]:
            diag.append(diagnostics(s,metric,t,x,start))
    dg=pd.DataFrame(diag);dg.to_csv(OUT/'convergence_diagnostics.csv',index=False)

    # Main convergence figure.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.9))
    ax=axes[0,0];panel(ax,'A')
    for s in SYSTEMS:
        x=brief[s];ax.plot(x.time_ns,x.protein_rmsd_A,color=COLORS[s],alpha=.18,lw=.4)
        ax.plot(x.time_ns,x.protein_rmsd_A.rolling(20,center=True,min_periods=1).mean(),color=COLORS[s],lw=1,label=s)
    ax.set(xlabel='Time (ns)',ylabel='Protein RMSD (Å)',title='Protein RMSD reaches a plateau');ax.legend(frameon=False,ncol=2);clean(ax)
    ax=axes[0,1];panel(ax,'B')
    for s in SYSTEMS:
        x=brief[s];ax.plot(x.time_ns,x.Mg_Mg_A.rolling(20,center=True,min_periods=1).mean(),color=COLORS[s],lw=1,label=s)
    ax.set(xlabel='Time (ns)',ylabel='Mg–Mg distance (Å)',title='Catalytic-metal geometry is stable');clean(ax)
    ax=axes[0,2];panel(ax,'C');blockrows=[]
    for s in SYSTEMS:
        x=pre[pre.system==s];bs=block_series(x.time_ns.to_numpy(),x.hybrid_eq39_basecentroid_A.to_numpy(),10,20)
        ax.plot([z[0] for z in bs],[z[1] for z in bs],'o-',color=COLORS[s],ms=3,lw=1,label=s)
        blockrows += [{'system':s,'block_mid_ns':t,'mean_A':v} for t,v in bs]
    ax.axhline(6.5,color='black',ls='--',lw=.7);ax.set(xlabel='10-ns block midpoint (ns)',ylabel='eq39–B25 distance (Å)',title='Block means separate after transition');clean(ax)
    pd.DataFrame(blockrows).to_csv(OUT/'eq39_10ns_block_means.csv',index=False)
    ax=axes[1,0];panel(ax,'D');cumrows=[]
    for s in SYSTEMS:
        x=pre[(pre.system==s)&(pre.time_ns>=30)];cum=x.hybrid_eq39_basecentroid_A.expanding().mean();ax.plot(x.time_ns,cum,color=COLORS[s],lw=1.2,label=s)
        cumrows += [{'system':s,'time_ns':t,'cumulative_mean_A':v} for t,v in zip(x.time_ns,cum)]
    ax.set(xlabel='Time (ns)',ylabel='Cumulative mean distance (Å)',title='Post-transition cumulative mean');clean(ax)
    pd.DataFrame(cumrows).to_csv(OUT/'eq39_cumulative_mean_from30ns.csv',index=False)
    ax=axes[1,1];panel(ax,'E');occrows=[]
    for s in SYSTEMS:
        x=pre[(pre.system==s)&(pre.time_ns>=30)];paired=((x.hybrid_eq39_basecentroid_A<6.5)&(x.hybrid_eq39_NO_contacts_3p5A>=1)).astype(float);cum=paired.expanding().mean()
        ax.plot(x.time_ns,cum,color=COLORS[s],lw=1.2,label=s);occrows += [{'system':s,'time_ns':t,'cumulative_paired_fraction':v} for t,v in zip(x.time_ns,cum)]
    ax.set(xlabel='Time (ns)',ylabel='Cumulative paired fraction',title='MM-Split does not re-pair after 30 ns',ylim=(-.03,1.03));clean(ax)
    pd.DataFrame(occrows).to_csv(OUT/'eq39_cumulative_pairing_from30ns.csv',index=False)
    ax=axes[1,2];panel(ax,'F');sens=[]
    for s in SYSTEMS:
        x=pre[pre.system==s];starts=np.arange(20,71,10);means=[x.loc[x.time_ns>=st,'hybrid_eq39_basecentroid_A'].mean() for st in starts]
        ax.plot(starts,means,'o-',color=COLORS[s],ms=3,lw=1,label=s);sens += [{'system':s,'discard_before_ns':st,'mean_A':v} for st,v in zip(starts,means)]
    ax.axhline(6.5,color='black',ls='--',lw=.7);ax.set(xlabel='Discard data before (ns)',ylabel='Retained-window mean (Å)',title='Conclusion is discard-window insensitive');clean(ax)
    pd.DataFrame(sens).to_csv(OUT/'eq39_discard_window_sensitivity.csv',index=False)
    fig.tight_layout();save(fig,'Figure_MD_convergence_main')

    # Supplement: explicitly display less-converged secondary observables and correlation time.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.9))
    ax=axes[0,0];panel(ax,'A')
    for s in SYSTEMS:
        x=brief[s];ax.plot(x.time_ns,x.pocket_rmsd_A.rolling(20,center=True,min_periods=1).mean(),color=COLORS[s],lw=1,label=s)
    ax.set(xlabel='Time (ns)',ylabel='Pocket RMSD (Å)',title='Pocket RMSD remains\ntime dependent');clean(ax)
    ax=axes[0,1];panel(ax,'B');volrows=[]
    for s in SYSTEMS:
        x=entry[entry.system==s];bs=block_series(x.time_ns.to_numpy(),x['protein_NTS_primary_accessible_volume_probe1.4_A3'].to_numpy(),10,20)
        ax.plot([z[0] for z in bs],[z[1] for z in bs],'o-',color=COLORS[s],ms=3,lw=1,label=s);volrows += [{'system':s,'block_mid_ns':t,'mean_A3':v} for t,v in bs]
    ax.set(xlabel='10-ns block midpoint (ns)',ylabel='NTS accessible volume (Å³)',title='Entry-volume block variation');clean(ax)
    pd.DataFrame(volrows).to_csv(OUT/'entry_volume_10ns_block_means.csv',index=False)
    ax=axes[0,2];panel(ax,'C')
    for s in SYSTEMS:
        x=pre[pre.system==s];bs=block_series(x.time_ns.to_numpy(),x.RuvCI_RuvCIII_COM_A.to_numpy(),10,20);ax.plot([z[0] for z in bs],[z[1] for z in bs],'o-',color=COLORS[s],ms=3,lw=1,label=s)
    ax.set(xlabel='10-ns block midpoint (ns)',ylabel='RuvC-I–III COM distance (Å)',title='RuvC separation by block');clean(ax)
    ax=axes[1,0];panel(ax,'D');acfrows=[]
    for s in SYSTEMS:
        x=pre[(pre.system==s)&(pre.time_ns>=30)].hybrid_eq39_basecentroid_A.to_numpy();ac,tau,ess=acf_tau(x);lag=np.arange(min(20,len(ac)));ax.plot(lag,ac[:len(lag)],color=COLORS[s],lw=1,label=f'{s} (ESS={ess:.0f})');acfrows += [{'system':s,'lag_ns':l,'acf':v} for l,v in zip(lag,ac[:len(lag)])]
    ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Lag (ns)',ylabel='Autocorrelation',title='eq39 autocorrelation and ESS');ax.legend(frameon=False,fontsize=5.8);clean(ax)
    pd.DataFrame(acfrows).to_csv(OUT/'eq39_autocorrelation.csv',index=False)
    ax=axes[1,1];panel(ax,'E');metrics=['protein_rmsd_A','pocket_rmsd_A','Mg_Mg_A','eq39_B25_basecentroid_A','RuvCI_RuvCIII_COM_A','NTS_volume_probe1p4_A3'];mat=dg.pivot(index='system',columns='metric',values='half_delta_in_SD').reindex(SYSTEMS)[metrics]
    im=ax.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-1.5,vmax=1.5);ax.set_xticks(range(len(metrics)),['Protein\nRMSD','Pocket\nRMSD','Mg–Mg','eq39–B25','RuvC\nI–III','Entry\nvolume'],rotation=35,ha='right');ax.set_yticks(range(4),SYSTEMS);ax.set_title('Half-window mean difference\n(SD units)');plt.colorbar(im,ax=ax,fraction=.045,pad=.03)
    ax=axes[1,2];panel(ax,'F');effect=[]
    mids=np.arange(35,100,10)
    for mid in mids:
        lo,hi=mid-5,mid+5;mm=pre[(pre.system=='MM-Split')&pre.time_ns.between(lo,hi,inclusive='left')].hybrid_eq39_basecentroid_A.mean();act=pre[(pre.system!='MM-Split')&pre.time_ns.between(lo,hi,inclusive='left')].hybrid_eq39_basecentroid_A.mean();effect.append(mm-act)
    ax.plot(mids,effect,'o-',color=COLORS['MM-Split'],lw=1.2);ax.axhline(0,color='black',lw=.7);ax.set(xlabel='10-ns block midpoint (ns)',ylabel='MM-Split − active-pool distance (Å)',title='Persistent blockwise separation');clean(ax)
    pd.DataFrame({'block_mid_ns':mids,'MMsplit_minus_active_pool_A':effect}).to_csv(OUT/'eq39_blockwise_effect.csv',index=False)
    fig.tight_layout();save(fig,'Figure_MD_convergence_supplement')

    summary={'scope':'internal convergence of one trajectory per condition','diagnostics':str(OUT/'convergence_diagnostics.csv'),
      'key_result':'After the 27-ns transition, MM-Split eq39-B25 remains unpaired through 100 ns; the post-30-ns mean, occupancy, block means, and discard-window sensitivity are stable and separated from all three wet-active systems.',
      'caution':'Protein RMSD and Mg geometry show stable plateaus, but pocket RMSD and entry volume retain time dependence in some systems. These data support robustness of the local eq39 finding, not full thermodynamic convergence.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

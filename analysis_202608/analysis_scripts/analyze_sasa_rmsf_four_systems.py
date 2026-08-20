#!/usr/bin/env python3
"""Focused four-system SASA and locally aligned RMSF analysis."""

from pathlib import Path
import sys, json, gc
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ruvc_entry_grid as ag

OUT=ROOT/'analysis_four_systems_20_100ns/sasa_rmsf_analysis'
PRE=ROOT/'analysis_four_systems_20_100ns/precatalytic_allostery/feature_timeseries.csv'
SYSTEMS=['Match-Full','Match-Split','MM-Full','MM-Split']
COLORS={'Match-Full':'#0072B2','Match-Split':'#009E73','MM-Full':'#E69F00','MM-Split':'#D55E00'}
BACKBONE={"P","OP1","OP2","OP3","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
SUGAR={"O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
PHOS={"P","OP1","OP2","OP3","O1P","O2P"}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})


def has_split(top):return any(c.chain_id=='D' and any(r.is_nucleic for r in c.residues) for c in top.chains)
def guide(top,eq):return ('D',eq-34) if has_split(top) and eq>=35 else ('C',eq)
def residue(top,ch,res):return next(r for r in top.residues if r.chain.chain_id==ch and r.resSeq==res)
def heavy_indices(res):return np.array([a.index for a in res.atoms if a.element and a.element.symbol!='H'],int)
def component(a):
    if a.name in PHOS:return 'phosphate'
    if a.name in SUGAR:return 'sugar'
    return 'base'


def isolated_sasa(tr,ids):
    q=tr.atom_slice(ids);return md.shrake_rupley(q,mode='atom',n_sphere_points=240)*100


def residue_rmsf(tr,residues):
    mean=tr.xyz.mean(0);disp=np.sum((tr.xyz-mean[None,:,:])**2,axis=2)
    rows=[]
    for label,res in residues:
        ids=heavy_indices(res);rows.append((label,float(np.sqrt(disp[:,ids].mean())*10)))
    return rows


def savefig(fig,name):
    for ext,dpi in [('png',600),('pdf',300),('svg',300)]:fig.savefig(OUT/f'{name}.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def panel(ax,label):ax.text(.015,.97,label,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.4))
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=3)
def blocks(x,n=8):return np.array([z.mean() for z in np.array_split(np.asarray(x),n)])


def main():
    OUT.mkdir(parents=True,exist_ok=True);sasa_rows=[];pair_rows=[];rmsf_rows=[]
    for system in SYSTEMS:
        tr,top,time=ag.load_system(system)
        selected=[]
        for r in range(24,30):selected.append((f'DNA_B{r}','B',r))
        for eq in range(35,45):
            ch,rr=guide(top,eq);selected.append((f'guide_eq{eq}',ch,rr))
        full=md.shrake_rupley(tr,mode='atom',n_sphere_points=240)*100
        iso_cache={}
        for label,ch,rn in selected:
            res=residue(top,ch,rn);ids=heavy_indices(res);iso=isolated_sasa(tr,ids);iso_total=iso.sum(1);iso_cache[label]=(ids,iso_total)
            comp_ids={c:np.array([a.index for a in res.atoms if a.element and a.element.symbol!='H' and component(a)==c],int) for c in ['base','sugar','phosphate']}
            for fi,t in enumerate(time):
                total=float(full[fi,ids].sum())
                sasa_rows.append({'system':system,'time_ns':t,'label':label,'chain':ch,'resSeq':rn,'resname':res.name,
                                  'component':'total','sasa_A2':total,'isolated_sasa_A2':float(iso_total[fi]),'relative_sasa':total/iso_total[fi]})
                for c,cids in comp_ids.items():
                    if len(cids):sasa_rows.append({'system':system,'time_ns':t,'label':label,'chain':ch,'resSeq':rn,'resname':res.name,
                      'component':c,'sasa_A2':float(full[fi,cids].sum()),'isolated_sasa_A2':np.nan,'relative_sasa':np.nan})
        for pname,glab,dlab in [('M17_eq38_B26','guide_eq38','DNA_B26'),('adjacent_eq39_B25','guide_eq39','DNA_B25')]:
            ids=np.r_[iso_cache[glab][0],iso_cache[dlab][0]];q=tr.atom_slice(ids);pair_sasa=md.shrake_rupley(q,mode='atom',n_sphere_points=240).sum(1)*100
            bsa=iso_cache[glab][1]+iso_cache[dlab][1]-pair_sasa
            for t,v,pv in zip(time,bsa,pair_sasa):pair_rows.append({'system':system,'time_ns':t,'pair':pname,'pair_BSA_A2':v,'pair_sasa_A2':pv})

        # Hybrid-local RMSF: align on stable flanks, excluding eq38/39 and B25/B26.
        q=tr[:]
        stable=[]
        for a in top.atoms:
            ch=a.residue.chain.chain_id;rr=a.residue.resSeq
            guide_eq=None
            if ch=='C':guide_eq=rr
            elif ch=='D' and has_split(top):guide_eq=rr+34
            dna_stable=(ch=='B' and (20<=rr<=24 or 27<=rr<=30))
            guide_stable=(guide_eq is not None and (35<=guide_eq<=37 or 40<=guide_eq<=44))
            if (dna_stable or guide_stable) and a.name in {'P',"C4'","C3'","C1'"}:stable.append(a.index)
        q.superpose(q,0,atom_indices=np.array(stable,int))
        rr=[]
        for r in range(24,30):rr.append((f'DNA_B{r}',residue(top,'B',r)))
        for eq in range(35,45):ch,rn=guide(top,eq);rr.append((f'guide_eq{eq}',residue(top,ch,rn)))
        for label,v in residue_rmsf(q,rr):rmsf_rows.append({'system':system,'alignment':'hybrid_flanks','label':label,'rmsf_A':v})

        # RuvC-local protein RMSF.
        qr=tr[:];ca=np.array([a.index for a in top.atoms if a.residue.chain.chain_id=='A' and a.name=='CA' and
             (809<=a.residue.resSeq<=872 or 891<=a.residue.resSeq<=997 or 1180<=a.residue.resSeq<=1226)],int);qr.superpose(qr,0,atom_indices=ca)
        pr=[]
        for r in list(range(809,998))+list(range(1180,1227)):
            try:pr.append((f'A_{r}',residue(top,'A',r)))
            except StopIteration:pass
        for label,v in residue_rmsf(qr,pr):rmsf_rows.append({'system':system,'alignment':'RuvC_CA','label':label,'rmsf_A':v})
        del tr,q,qr,full;gc.collect()

    sasa=pd.DataFrame(sasa_rows);pairs=pd.DataFrame(pair_rows);rmsf=pd.DataFrame(rmsf_rows)
    sasa.to_csv(OUT/'sasa_timeseries_long.csv',index=False);pairs.to_csv(OUT/'basepair_bsa_timeseries.csv',index=False);rmsf.to_csv(OUT/'local_rmsf_by_residue.csv',index=False)
    summary=(sasa.groupby(['system','label','component']).agg(mean_sasa_A2=('sasa_A2','mean'),sd_sasa_A2=('sasa_A2','std'),
             mean_relative_sasa=('relative_sasa','mean')).reset_index());summary.to_csv(OUT/'sasa_summary.csv',index=False)
    psum=pairs.groupby(['system','pair']).pair_BSA_A2.agg(['mean','std','median']).reset_index();psum.to_csv(OUT/'basepair_bsa_summary.csv',index=False)

    # Publication figure.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.9));pre=pd.read_csv(PRE)
    for ax,label,lab,title in [(axes[0,0],'guide_eq39','A','Guide eq39 base exposure'),(axes[0,1],'DNA_B25','B','Target B25 base exposure')]:
        panel(ax,lab)
        for s in SYSTEMS:
            x=sasa[(sasa.system==s)&(sasa.label==label)&(sasa.component=='base')];ax.plot(x.time_ns,x.sasa_A2.rolling(3,center=True,min_periods=1).mean(),color=COLORS[s],lw=1,label=s)
        ax.set(xlabel='Time (ns)',ylabel='Base SASA (Å²)',title=title);clean(ax)
    axes[0,0].legend(frameon=False,ncol=2)
    ax=axes[0,2];panel(ax,'C')
    for s in SYSTEMS:
        x=pairs[(pairs.system==s)&(pairs.pair=='adjacent_eq39_B25')];bs=blocks(x.pair_BSA_A2);ax.scatter(np.full(8,SYSTEMS.index(s))+np.linspace(-.09,.09,8),bs,color=COLORS[s],s=15,alpha=.8);ax.plot([SYSTEMS.index(s)-.16,SYSTEMS.index(s)+.16],[bs.mean()]*2,color='black',lw=1.2)
    ax.set_xticks(range(4),['Match\nFull','Match\nSplit','MM\nFull','MM\nSplit']);ax.set_ylabel('eq39–B25 pair BSA (Å²)');ax.set_title('Loss of inter-base burial');clean(ax)
    ax=axes[1,0];panel(ax,'D')
    for s in SYSTEMS:
        x=rmsf[(rmsf.system==s)&(rmsf.alignment=='hybrid_flanks')&rmsf.label.str.startswith('guide_eq')].copy();x['pos']=x.label.str.extract(r'(\d+)').astype(int);x=x[x.pos.between(37,40)];ax.plot(x.pos,x.rmsf_A,'o-',color=COLORS[s],ms=3,lw=1,label=s)
    ax.axvspan(37.7,39.3,color='0.85',zorder=0);ax.set(xlabel='Equivalent guide position',ylabel='Locally aligned RMSF (Å)',title='Guide local flexibility');clean(ax)
    ax=axes[1,1];panel(ax,'E')
    for s in SYSTEMS:
        x=rmsf[(rmsf.system==s)&(rmsf.alignment=='hybrid_flanks')&rmsf.label.str.startswith('DNA_B')].copy();x['pos']=x.label.str.extract(r'(\d+)').astype(int);ax.plot(x.pos,x.rmsf_A,'o-',color=COLORS[s],ms=3,lw=1,label=s)
    ax.axvspan(24.7,26.3,color='0.85',zorder=0);ax.set(xlabel='Target-DNA residue',ylabel='Locally aligned RMSF (Å)',title='Target-DNA local flexibility');clean(ax)
    ax=axes[1,2];panel(ax,'F');rows=[]
    for s in SYSTEMS:
        a=sasa[(sasa.system==s)&(sasa.label.isin(['guide_eq39','DNA_B25']))&(sasa.component=='base')].pivot(index='time_ns',columns='label',values='sasa_A2').sum(1)
        d=pre[pre.system==s].set_index('time_ns').hybrid_eq39_basecentroid_A.reindex(a.index);ax.scatter(d,a,s=10,color=COLORS[s],alpha=.55,label=s);rows += [{'system':s,'distance_A':xx,'combined_base_sasa_A2':yy} for xx,yy in zip(d,a)]
    ax.set(xlabel='eq39–B25 distance (Å)',ylabel='Combined base SASA (Å²)',title='Exposure coupled to dissociation');clean(ax)
    pd.DataFrame(rows).to_csv(OUT/'eq39_distance_vs_combined_base_sasa.csv',index=False)
    fig.tight_layout();savefig(fig,'Figure_MD_SASA_RMSF_main')

    # Position-resolved supplementary heat maps.
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5))
    for ax,prefix,positions,comp,lab,title in [(axes[0,0],'guide_eq',range(35,45),'base','A','Guide base SASA'),
      (axes[0,1],'DNA_B',range(24,30),'base','B','Target-DNA base SASA'),(axes[1,0],'guide_eq',range(35,45),'total','C','Guide relative SASA'),
      (axes[1,1],'DNA_B',range(24,30),'total','D','Target-DNA relative SASA')]:
        panel(ax,lab);mat=[]
        for s in SYSTEMS:
            vals=[]
            for p in positions:
                x=sasa[(sasa.system==s)&(sasa.label==f'{prefix}{p}')&(sasa.component==comp)]
                vals.append(x.relative_sasa.mean() if comp=='total' else x.sasa_A2.mean())
            mat.append(vals)
        im=ax.imshow(mat,aspect='auto',cmap='magma');ax.set_xticks(range(len(list(positions))),list(positions));ax.set_yticks(range(4),SYSTEMS);ax.set_title(title);plt.colorbar(im,ax=ax,fraction=.045,pad=.03)
    fig.tight_layout();savefig(fig,'Figure_MD_SASA_RMSF_supplement')

    out={'method':{'frames_per_system':80,'window_ns':'20-100','sampling_ns':1,'sasa':'Shrake-Rupley, 240 sphere points, solute geometric SASA',
      'relative_sasa':'full-complex nucleotide SASA divided by same-frame isolated-nucleotide SASA',
      'hybrid_rmsf_alignment':'stable hybrid flanks excluding guide eq38/39 and DNA B25/B26',
      'RuvC_rmsf_alignment':'RuvC CA residues 809-872, 891-997, 1180-1226'},
      'limitations':['one trajectory per condition','SASA uses solute geometry; water coordinates are not required','RMSF is alignment- and time-window-dependent']}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()

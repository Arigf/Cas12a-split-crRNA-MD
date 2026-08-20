#!/usr/bin/env python3
"""Create publication-ready main and supplementary figures for the four MD systems."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'analysis_four_systems_20_100ns/precatalytic_allostery'
ENT=ROOT/'analysis_four_systems_20_100ns/ruvc_entry_grid_analysis'
PROD=ROOT/'production_md_2p5mM_MgCl2_37p5mM_KCl'
OUT=ROOT/'analysis_four_systems_20_100ns/publication_figures'
DATA=OUT/'source_data'
SYSTEMS=['Match-Full','Match-Split','MM-Full','MM-Split']
COLORS={'Match-Full':'#0072B2','Match-Split':'#009E73','MM-Full':'#E69F00','MM-Split':'#D55E00'}

mpl.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.8,
    'xtick.major.width':0.7,'ytick.major.width':0.7,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.bbox':'tight','figure.facecolor':'white'
})


def panel(ax,label):
    ax.text(-0.20,0.99,label,transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3)


def blocks(x,n=8): return np.array([v.mean() for v in np.array_split(np.asarray(x),n)])


def block_dot(ax,df,col,ylabel,zero=None):
    rng=np.random.default_rng(20260803)
    rows=[]
    for i,s in enumerate(SYSTEMS):
        vals=blocks(df.loc[df.system==s,col])
        rows += [{'system':s,'block':j+1,'value':v} for j,v in enumerate(vals)]
        jitter=rng.uniform(-.10,.10,len(vals))
        ax.scatter(i+jitter,vals,s=18,color=COLORS[s],alpha=.75,edgecolor='white',linewidth=.35,zorder=2)
        ax.plot([i-.17,i+.17],[vals.mean(),vals.mean()],color='black',lw=1.4,zorder=3)
    if zero is not None: ax.axhline(zero,color='0.45',ls='--',lw=.8)
    ax.set_xticks(range(4),['Match\nFull','Match\nSplit','MM\nFull','MM\nSplit'])
    ax.set_ylabel(ylabel);clean(ax)
    return pd.DataFrame(rows)


def save(fig,name):
    for ext,dpi in [('png',600),('pdf',300),('svg',300)]: fig.savefig(OUT/f'{name}.{ext}',dpi=dpi)
    plt.close(fig)


def construct_schematic(ax):
    ax.set_xlim(30,45);ax.set_ylim(-.7,3.7);ax.axis('off')
    for row,s in enumerate(SYSTEMS[::-1]):
        y=row
        ax.plot([31,44],[y+.13,y+.13],color='0.35',lw=2.4,solid_capstyle='round')
        if 'Split' in s:
            ax.plot([31,34],[y-.13,y-.13],color='#56B4E9',lw=4,solid_capstyle='round')
            ax.plot([35,44],[y-.13,y-.13],color='#CC79A7',lw=4,solid_capstyle='round')
            ax.plot([34.35,34.65],[y-.30,y+.02],color='black',lw=1)
        else:
            ax.plot([31,44],[y-.13,y-.13],color='#56B4E9',lw=4,solid_capstyle='round')
        if s.startswith('MM'):
            ax.text(38,y+.02,'×',color='#D55E00',ha='center',va='center',fontsize=12,fontweight='bold')
        ax.scatter([39],[y-.13],s=25,color='#CC79A7',edgecolor='black',linewidth=.4,zorder=4)
        ax.text(30.7,y,s,ha='right',va='center',fontsize=7)
    ax.text(35.4,3.47,'M17 / eq38–B26',ha='center',fontsize=7,color='#D55E00')
    ax.annotate('',xy=(38,3.15),xytext=(36.5,3.40),arrowprops=dict(arrowstyle='-',lw=.7,color='#D55E00'))
    ax.text(41.8,3.47,'eq39–B25',ha='center',fontsize=7,color='#AA3377')
    ax.annotate('',xy=(39,3.15),xytext=(41.2,3.40),arrowprops=dict(arrowstyle='-',lw=.7,color='#AA3377'))
    ax.text(31,-.53,'guide 31',ha='center',fontsize=6);ax.text(44,-.53,'guide 44',ha='center',fontsize=6)
    ax.set_title('Four simulated complexes',loc='left',pad=1)


def main_figure(pre,entry):
    fig=plt.figure(figsize=(7.2,7.0));gs=GridSpec(3,2,figure=fig,hspace=.48,wspace=.34)
    ax=fig.add_subplot(gs[0,0]);construct_schematic(ax);panel(ax,'A')

    ax=fig.add_subplot(gs[0,1]);panel(ax,'B')
    for s in SYSTEMS:
        x=pre[pre.system==s]
        ax.plot(x.time_ns,x.hybrid_eq39_basecentroid_A,color=COLORS[s],alpha=.22,lw=.6)
        ax.plot(x.time_ns,x.hybrid_eq39_basecentroid_A.rolling(3,center=True,min_periods=1).mean(),color=COLORS[s],lw=1.3,label=s)
    ax.axhline(6.5,color='black',ls='--',lw=.8,label='paired cutoff')
    ax.set(xlabel='Time (ns)',ylabel='eq39–B25 base-centroid distance (Å)',title='Persistent adjacent-pair loss in MM-Split');ax.set_xlim(20,99);ax.legend(ncol=2,frameon=False);clean(ax)

    ax=fig.add_subplot(gs[1,0]);panel(ax,'C')
    occ=np.zeros((4,10))
    for i,s in enumerate(SYSTEMS):
        x=pre[pre.system==s]
        for j,eq in enumerate(range(35,45)):
            occ[i,j]=np.mean((x[f'hybrid_eq{eq}_basecentroid_A']<6.5)&(x[f'hybrid_eq{eq}_NO_contacts_3p5A']>=1))
    im=ax.imshow(occ,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(10),range(35,45));ax.set_yticks(range(4),SYSTEMS);ax.set(xlabel='Equivalent guide position',title='RNA–DNA paired-state occupancy')
    ax.axvline(3-.5,color='white',lw=.8,ls=':');ax.axvline(4-.5,color='white',lw=.8,ls=':')
    for i in range(4):
        for j in range(10):
            if j in (3,4): ax.text(j,i,f'{occ[i,j]*100:.0f}',ha='center',va='center',fontsize=5.8,color=('white' if occ[i,j]<.45 else 'black'))
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label('Fraction')
    pd.DataFrame(occ,index=SYSTEMS,columns=[f'eq{x}' for x in range(35,45)]).to_csv(DATA/'main_C_pairing_occupancy.csv')

    ax=fig.add_subplot(gs[1,1]);panel(ax,'D')
    src=block_dot(ax,pre,'hybrid_eq39_NO_contacts_3p5A','eq39–B25 N/O contacts (<3.5 Å)')
    ax.set_title('Loss of base-pair contacts');src.to_csv(DATA/'main_D_eq39_NO_contact_blocks.csv',index=False)

    ax=fig.add_subplot(gs[2,0]);panel(ax,'E')
    src=block_dot(ax,entry,'protein_NTS_primary_accessible_volume_probe1.4_A3','NTS-path accessible volume (Å³)')
    ax.set_title('MM-Split is not entrance-occluded');src.to_csv(DATA/'main_E_NTS_volume_blocks.csv',index=False)

    ax=fig.add_subplot(gs[2,1]);panel(ax,'F')
    src=block_dot(ax,pre,'RuvCI_RuvCIII_COM_A','RuvC-I–RuvC-III COM distance (Å)')
    ax.set_title('Modest RuvC lobe expansion');src.to_csv(DATA/'main_F_RuvCI_III_blocks.csv',index=False)
    fig.align_ylabels();save(fig,'Figure_MD_main')


def stability_figure(pre):
    brief=[]
    for s in SYSTEMS:
        x=pd.read_csv(PROD/s/'analysis_brief/timeseries_20_100ns.csv');x['system']=s;brief.append(x)
    d=pd.concat(brief,ignore_index=True);d.to_csv(DATA/'supp1_MD_quality_timeseries.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8));metrics=[
        ('protein_rmsd_A','Protein RMSD (Å)','A'),('pocket_rmsd_A','Catalytic-pocket RMSD (Å)','B'),
        ('Mg_Mg_A','Mg–Mg distance (Å)','C')]
    for ax,(col,label,lab) in zip(axes[0],metrics):
        panel(ax,lab)
        for s in SYSTEMS:
            x=d[d.system==s];ax.plot(x.time_ns,x[col],color=COLORS[s],lw=.7,alpha=.8,label=s)
        ax.set(xlabel='Time (ns)',ylabel=label);clean(ax)
    axes[0,0].legend(ncol=2,frameon=False)
    src=block_dot(axes[1,0],pre,'hybrid_eq38_basecentroid_A','M17/eq38 base-centroid distance (Å)');panel(axes[1,0],'D');axes[1,0].set_title('M17 geometry');src.to_csv(DATA/'supp1_D_eq38_blocks.csv',index=False)
    src=block_dot(axes[1,1],pre,'guide34_O3_guide35_O5_A',"guide34 O3′–guide35 O5′ (Å)");panel(axes[1,1],'E');axes[1,1].set_title('Split-junction geometry');src.to_csv(DATA/'supp1_E_split_break_blocks.csv',index=False)
    src=block_dot(axes[1,2],pre,'guide_stack_eq38_39_centroid_A','eq38–eq39 guide stacking (Å)');panel(axes[1,2],'F');axes[1,2].set_title('Guide stacking retained');src.to_csv(DATA/'supp1_F_guide_stack_blocks.csv',index=False)
    fig.tight_layout();save(fig,'Figure_MD_supp1_stability_local')


def entry_figure(entry):
    prof=pd.read_csv(ENT/'path_clearance_profiles_summary.csv');prof.to_csv(DATA/'supp2_path_clearance_profiles.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8))
    for ax,path,lab,title in [(axes[0,0],'NTS_primary','A','NTS reference path'),(axes[0,1],'TS_sensitivity','B','TS sensitivity path')]:
        panel(ax,lab)
        for s in SYSTEMS:
            x=prof[(prof.system==s)&(prof.path==path)]
            ax.plot(x.arclength_A,x.mean_A,color=COLORS[s],lw=1,label=s)
            ax.fill_between(x.arclength_A,x.p05_A,x.p95_A,color=COLORS[s],alpha=.10)
        ax.axhline(1.4,color='black',ls='--',lw=.7);ax.set(xlabel='Path arclength (Å)',ylabel='Surface clearance (Å)',title=title);clean(ax)
    axes[0,0].legend(ncol=2,frameon=False)
    ax=axes[0,2];panel(ax,'C');src=block_dot(ax,entry,'protein_NTS_primary_min_clearance_A','Minimum clearance (Å)');ax.set_title('NTS bottleneck');src.to_csv(DATA/'supp2_C_NTS_clearance_blocks.csv',index=False)
    ax=axes[1,0];panel(ax,'D');src=block_dot(ax,entry,'protein_NTS_primary_accessible_volume_probe2.5_A3','Accessible volume (Å³)');ax.set_title('NTS, 2.5-Å probe');src.to_csv(DATA/'supp2_D_NTS_2p5_volume_blocks.csv',index=False)
    ax=axes[1,1];panel(ax,'E');src=block_dot(ax,entry,'protein_TS_sensitivity_accessible_volume_probe1.4_A3','Accessible volume (Å³)');ax.set_title('TS path, 1.4-Å probe');src.to_csv(DATA/'supp2_E_TS_volume_blocks.csv',index=False)
    ax=axes[1,2];panel(ax,'F')
    means=[]
    for i,s in enumerate(SYSTEMS):
        x=entry[entry.system==s]
        pv=x['protein_NTS_primary_accessible_volume_probe1.4_A3'].mean();cv=x['complex_NTS_primary_accessible_volume_probe1.4_A3'].mean();means.append([s,pv,cv])
        ax.bar(i-.17,pv,width=.34,color=COLORS[s],alpha=.9);ax.bar(i+.17,cv,width=.34,color=COLORS[s],alpha=.35,hatch='//')
    ax.set_xticks(range(4),['Match\nFull','Match\nSplit','MM\nFull','MM\nSplit']);ax.set_ylabel('Accessible volume (Å³)');ax.set_title('Protein wall vs\noccupied complex',fontsize=8.5);clean(ax)
    ax.plot([],[],color='0.3',lw=6,label='protein');ax.bar([],[],color='0.7',hatch='//',label='+ resident nucleic acids');ax.legend(frameon=False,fontsize=6)
    pd.DataFrame(means,columns=['system','protein_wall_A3','complex_wall_A3']).to_csv(DATA/'supp2_F_wall_mode_means.csv',index=False)
    fig.tight_layout();save(fig,'Figure_MD_supp2_entry_bottleneck')


def network_figure(pre):
    net=pd.read_csv(PRE/'allosteric_network_paths.csv');state=pd.read_csv(PRE/'state_occupancy.csv');load=pd.read_csv(PRE/'pca_feature_loadings.csv',index_col=0)
    net.to_csv(DATA/'supp3_allosteric_network_paths.csv',index=False);state.to_csv(DATA/'supp3_state_occupancy.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7))
    ax=axes[0];panel(ax,'A');width=.18
    targets=['A:832','A:925','A:1180']
    for i,s in enumerate(SYSTEMS):
        vals=[net[(net.system==s)&(net.source=='G:35')&(net.target==t)].cost.iloc[0] for t in targets]
        ax.bar(np.arange(3)+(i-1.5)*width,vals,width,color=COLORS[s],label=s)
    ax.set_xticks(range(3),['D832','E925','D1180']);ax.set_ylabel('Network path cost');ax.set_title('G35-to-catalytic network',fontsize=8.5);clean(ax);ax.legend(frameon=False,fontsize=6)
    ax=axes[1];panel(ax,'B');mat=state.set_index('system').reindex(SYSTEMS)[[str(i) for i in range(4)]].to_numpy();im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(4),['S0','S1','S2','S3']);ax.set_yticks(range(4),SYSTEMS);ax.set_title('Unsupervised state occupancy',fontsize=8.5);fig.colorbar(im,ax=ax,fraction=.05,pad=.03)
    ax=axes[2];panel(ax,'C');top=load.PC1.abs().nlargest(10).index[::-1];vals=load.loc[top,'PC1'];ax.barh(range(len(top)),vals,color=np.where(vals>0,'#0072B2','#D55E00'));ax.set_yticks(range(len(top)),[x.replace('_A','') for x in top],fontsize=5.5);ax.set_xlabel('PC1 loading');ax.set_title('Dominant state-space features',fontsize=8.5);clean(ax)
    fig.tight_layout();save(fig,'Figure_MD_supp3_network_states')


def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    pre=pd.read_csv(PRE/'feature_timeseries.csv');entry=pd.read_csv(ENT/'timeseries_20_100ns_1ns.csv')
    pre.to_csv(DATA/'precatalytic_feature_timeseries.csv',index=False);entry.to_csv(DATA/'entry_grid_timeseries.csv',index=False)
    main_figure(pre,entry);stability_figure(pre);entry_figure(entry);network_figure(pre)
    manifest={'figures':['Figure_MD_main','Figure_MD_supp1_stability_local','Figure_MD_supp2_entry_bottleneck','Figure_MD_supp3_network_states'],
              'formats':['png','pdf','svg'],'systems':SYSTEMS,'window_ns':'20-100','sampling':'1 ns for pre-catalytic/entry panels; 0.1 ns traces for stability panels',
              'statistics':'eight contiguous 10-ns block means shown as descriptive dots; blocks and frames are not independent MD replicas'}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()

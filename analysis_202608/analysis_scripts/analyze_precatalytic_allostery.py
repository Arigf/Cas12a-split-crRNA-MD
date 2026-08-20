#!/usr/bin/env python3
"""Wet-label-guided pre-catalytic geometry, state, and allosteric analysis."""

from __future__ import annotations
import argparse, json, sys, gc
from pathlib import Path

import mdtraj as md
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ruvc_entry_grid as ag

OUT=ROOT/'analysis_four_systems_20_100ns/precatalytic_allostery'
WET_ACTIVE={'Match-Full':1,'Match-Split':1,'MM-Full':1,'MM-Split':0}
CLEAVAGE_RESIDUES=range(14,25)


def atom(top,ch,res,name):
    return next(a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res and a.name==name)


def maybe_atom(top,ch,res,name):
    return next((a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res and a.name==name), None)


def residue_atoms(top,ch,res,heavy=True):
    return np.array([a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res
                     and (not heavy or (a.element and a.element.symbol!='H'))],int)


def base_atoms(top,ch,res):
    backbone={"P","OP1","OP2","OP3","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
    return np.array([a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res
                     and a.name not in backbone and a.element and a.element.symbol!='H'],int)


def dist(x,a,b): return np.linalg.norm(x[:,a]-x[:,b],axis=1)
def com(x,ids): return x[:,ids].mean(axis=1)
def comdist(x,a,b): return np.linalg.norm(com(x,a)-com(x,b),axis=1)


def angle3(a,b,c):
    u=a-b;v=c-b
    co=np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
    return np.degrees(np.arccos(np.clip(co,-1,1)))


def global_reference(reference_pdb=None):
    import mdtraj as md
    reference_pdb=reference_pdb or ag.EQ/'Match-Full'/'06_unrestrained_stability_npt.pdb'
    t=md.load_pdb(str(reference_pdb))
    return {(a.residue.resSeq,a.name):t.xyz[0,a.index]*10 for a in t.topology.atoms
            if a.residue.chain.chain_id=='A' and 1<=a.residue.resSeq<=1230 and a.name=='CA'}


def align_global(tr,top,ref):
    ids=[];target=[]
    for a in top.atoms:
        key=(a.residue.resSeq,a.name)
        if a.residue.chain.chain_id=='A' and a.name=='CA' and key in ref:
            ids.append(a.index);target.append(ref[key])
    ids=np.asarray(ids);target=np.asarray(target);x=tr.xyz*10
    for f in range(len(x)):
        r,t=ag.kabsch(x[f,ids],target);x[f]=x[f]@r+t
    return x


def equivalent_guide(top,eq_res):
    has_split_back=any(c.chain_id=='D' and any(r.is_nucleic for r in c.residues) for c in top.chains)
    if has_split_back and eq_res>=35: return 'D',eq_res-34
    return 'C',eq_res


def load_completed_rep1(system):
    base=ROOT/'precatalytic_production_100ns_3rep'/system/'rep1'
    prefix=base/'prod_rep1_100ns'
    idx=np.loadtxt(str(prefix)+'_solute_atom_indices.txt',dtype=int)
    full=md.load_pdb(str(ROOT/'precatalytic_four_systems_staged_equilibration'/system/'06_unrestrained_stability_npt.pdb'))
    top=full.atom_slice(idx).topology
    del full
    ag.repair(top)
    tr=md.load(str(prefix)+'_solute.dcd',top=top)
    molecules=list(top.find_molecules())
    anchor=max(molecules,key=len)
    tr=tr.image_molecules(anchor_molecules=[anchor],other_molecules=[m for m in molecules if m is not anchor],make_whole=True)
    # Reporter starts from a state at 70 ps; DCD cadence is 100 ps.
    return tr,top,0.07+np.arange(tr.n_frames)*0.1


def contact_count(x,a,b,cut=4.5):
    out=[]
    for frame in x:
        # Small selections; broadcasting is faster than constructing all atom pairs.
        d=np.linalg.norm(frame[a,None,:]-frame[None,b,:],axis=2)
        out.append(np.count_nonzero(d<cut))
    return np.asarray(out)


def feature_table(system,tr,top,x,time):
    f=pd.DataFrame({'system':system,'time_ns':time,'wet_active':WET_ACTIVE[system]})
    mg=np.array([a.index for a in top.atoms if a.element and a.element.symbol=='Mg']); assert len(mg)==2
    m1,m2=mg
    # Metal/catalytic geometry.
    f['Mg_Mg_A']=dist(x,m1,m2)
    for label,mi,res,names in [('Mg1_D832',m1,832,['OD1','OD2']),('Mg1_E925',m1,925,['OE1','OE2']),
                               ('Mg2_D832',m2,832,['OD1','OD2']),('Mg2_D1180',m2,1180,['OD1','OD2'])]:
        f[label+'_A']=np.min(np.column_stack([dist(x,mi,atom(top,'A',res,n)) for n in names]),axis=1)
    ps=[a for a in top.atoms if a.residue.chain.chain_id=='B' and a.name=='P']
    pmat=np.column_stack([np.minimum(dist(x,m1,a.index),dist(x,m2,a.index)) for a in ps])
    nearest=np.argmin(pmat,axis=1)
    f['DNA_anyP_to_Mg_min_A']=pmat[np.arange(len(x)),nearest]
    f['DNA_nearest_P_residue']=[ps[i].residue.resSeq for i in nearest]
    p18=atom(top,'B',18,'P');op18=[atom(top,'B',18,n) for n in ['OP1','OP2']]
    f['B18_P_to_Mg_min_A']=np.minimum(dist(x,m1,p18),dist(x,m2,p18))
    f['B18_OP_to_Mg_min_A']=np.min(np.column_stack([dist(x,m,o) for m in mg for o in op18]),axis=1)
    o3=atom(top,'B',17,"O3'")
    mgc=x[:,mg].mean(1)
    f['B18_Mgcentroid_P_O3_angle_deg']=angle3(mgc,x[:,p18],x[:,o3])

    # RuvC piece geometry selections used below.
    ranges={'RuvCI':(809,872),'RuvCII':(891,997),'RuvCIII':(1180,1226),'BH':(870,890),'lid':(926,941)}
    ids={k:np.concatenate([residue_atoms(top,'A',r) for r in range(lo,hi+1)]) for k,(lo,hi) in ranges.items()}
    cat=np.concatenate([residue_atoms(top,'A',r) for r in [832,925,1180]])

    # Cleavage-strand geometry around the common B18 scissile phosphate.
    # Chain B is the common substrate strand in the B18 precatalytic template.
    f['cut_B18_P_to_Mgcentroid_A']=np.linalg.norm(x[:,p18]-mgc,axis=1)
    for name in ['OP1','OP2']:
        oi=atom(top,'B',18,name)
        f[f'cut_B18_{name}_to_Mg1_A']=dist(x,oi,m1)
        f[f'cut_B18_{name}_to_Mg2_A']=dist(x,oi,m2)
        f[f'cut_B18_{name}_to_Mg_min_A']=np.minimum(dist(x,oi,m1),dist(x,oi,m2))
    f['cut_B18_OP_to_Mg_min_A']=np.minimum(f['cut_B18_OP1_to_Mg_min_A'],f['cut_B18_OP2_to_Mg_min_A'])
    o5=maybe_atom(top,'B',18,"O5'")
    if o5 is not None:
        f['cut_B18_O5_to_Mg_min_A']=np.minimum(dist(x,o5,m1),dist(x,o5,m2))
    p17=maybe_atom(top,'B',17,'P');p19=maybe_atom(top,'B',19,'P')
    if p17 is not None and p19 is not None:
        f['cut_B17P_B18P_B19P_angle_deg']=angle3(x[:,p17],x[:,p18],x[:,p19])
        f['cut_B17P_B18P_A']=dist(x,p17,p18)
        f['cut_B18P_B19P_A']=dist(x,p18,p19)
    cut_heavy=np.concatenate([residue_atoms(top,'B',r) for r in CLEAVAGE_RESIDUES if len(residue_atoms(top,'B',r))])
    f['cut_B14_B24_RuvC_contacts_4p5A']=contact_count(x,cut_heavy,np.concatenate([ids['RuvCI'],ids['RuvCII'],ids['RuvCIII']]))
    f['cut_B14_B24_lid_contacts_4p5A']=contact_count(x,cut_heavy,ids['lid'])
    f['cut_B14_B24_RuvC_COM_A']=comdist(x,cut_heavy,np.concatenate([ids['RuvCI'],ids['RuvCII'],ids['RuvCIII']]))
    cut_p_atoms=[]
    cut_p_res=[]
    for res in CLEAVAGE_RESIDUES:
        pi=maybe_atom(top,'B',res,'P')
        if pi is None:
            continue
        cut_p_atoms.append(pi);cut_p_res.append(res)
        f[f'cut_B{res}_P_to_Mg_min_A']=np.minimum(dist(x,pi,m1),dist(x,pi,m2))
        f[f'cut_B{res}_P_to_Mgcentroid_A']=np.linalg.norm(x[:,pi]-mgc,axis=1)
        ops=[maybe_atom(top,'B',res,n) for n in ['OP1','OP2']]
        ops=[o for o in ops if o is not None]
        if ops:
            f[f'cut_B{res}_OP_to_Mg_min_A']=np.min(np.column_stack([dist(x,m,o) for m in mg for o in ops]),axis=1)
    if cut_p_atoms:
        cut_p_mat=np.column_stack([np.minimum(dist(x,m1,pi),dist(x,m2,pi)) for pi in cut_p_atoms])
        nearest_cut=np.argmin(cut_p_mat,axis=1)
        f['cut_nearest_B14_B24_P_to_Mg_A']=cut_p_mat[np.arange(len(x)),nearest_cut]
        f['cut_nearest_B14_B24_P_residue']=[cut_p_res[i] for i in nearest_cut]

    # RuvC piece geometry, lid/BH conformations, and entry-network side chains.
    for a,b in [('RuvCI','RuvCII'),('RuvCI','RuvCIII'),('RuvCII','RuvCIII'),('BH','lid'),('BH','RuvCIII'),('lid','RuvCIII')]:
        f[f'{a}_{b}_COM_A']=comdist(x,ids[a],ids[b])
    f['lid_catalytic_COM_A']=comdist(x,ids['lid'],cat);f['BH_catalytic_COM_A']=comdist(x,ids['BH'],cat)
    bh1=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(871,877)]))
    bh2=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(877,884)]))
    bh3=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(884,890)]))
    f['BH_bend_angle_deg']=angle3(bh1,bh2,bh3)
    lid1=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(927,932)]))
    lid2=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(932,936)]))
    lid3=com(x,np.concatenate([residue_atoms(top,'A',r) for r in range(936,941)]))
    f['lid_bend_angle_deg']=angle3(lid1,lid2,lid3)
    for a,b in [(999,1084),(999,1138),(1084,1138),(1084,1142),(1138,1210),(1142,1210)]:
        f[f'A{a}_A{b}_COM_A']=comdist(x,residue_atoms(top,'A',a),residue_atoms(top,'A',b))

    # Split-equivalent guide break and distal hybrid.
    c34,r34=equivalent_guide(top,34);c35,r35=equivalent_guide(top,35)
    f['guide34_O3_guide35_O5_A']=dist(x,atom(top,c34,r34,"O3'"),atom(top,c35,r35,"O5'"))
    back=np.concatenate([residue_atoms(top,*equivalent_guide(top,r)) for r in range(35,45)])
    f['guide_back_lid_COM_A']=comdist(x,back,ids['lid']);f['guide_back_BH_COM_A']=comdist(x,back,ids['BH'])
    f['guide_back_RuvC_contacts_4p5A']=contact_count(x,back,np.concatenate([ids['RuvCI'],ids['RuvCII'],ids['RuvCIII']]))
    dna_distal=np.concatenate([residue_atoms(top,'B',r) for r in range(20,30)])
    f['guide_back_DNA_contacts_4p5A']=contact_count(x,back,dna_distal)
    for eq in range(35,45):
        ch,rr=equivalent_guide(top,eq); br=64-eq
        ga=base_atoms(top,ch,rr);da=base_atoms(top,'B',br)
        f[f'hybrid_eq{eq}_basecentroid_A']=comdist(x,ga,da)
        bd=np.linalg.norm(x[:,ga,None,:]-x[:,None,da,:],axis=3)
        f[f'hybrid_eq{eq}_min_base_A']=bd.min(axis=(1,2))
        f[f'hybrid_eq{eq}_base_contacts_3p5A']=(bd<3.5).sum(axis=(1,2))
        gp=np.array([i for i in ga if list(top.atoms)[i].element.symbol in {'N','O'}],int)
        dp=np.array([i for i in da if list(top.atoms)[i].element.symbol in {'N','O'}],int)
        hd=np.linalg.norm(x[:,gp,None,:]-x[:,None,dp,:],axis=3)
        f[f'hybrid_eq{eq}_NO_contacts_3p5A']=(hd<3.5).sum(axis=(1,2))
    for left,right in [(37,38),(38,39),(39,40)]:
        lc,lr=equivalent_guide(top,left);rc,rr=equivalent_guide(top,right)
        f[f'guide_stack_eq{left}_{right}_centroid_A']=comdist(x,base_atoms(top,lc,lr),base_atoms(top,rc,rr))
        f[f'DNA_stack_B{64-left}_B{64-right}_centroid_A']=comdist(x,base_atoms(top,'B',64-left),base_atoms(top,'B',64-right))
    return f


def network(system,top,x):
    # Protein CA nodes plus guide phosphate/C4' representatives in the allosteric corridor.
    nodes=[];coords=[]
    for r in range(800,1227):
        try:i=atom(top,'A',r,'CA')
        except StopIteration:continue
        nodes.append(f'A:{r}');coords.append(x[:,i])
    for eq in range(31,45):
        ch,rr=equivalent_guide(top,eq)
        try:i=atom(top,ch,rr,'P')
        except StopIteration:i=atom(top,ch,rr,"C4'")
        nodes.append(f'G:{eq}');coords.append(x[:,i])
    pos=np.stack(coords,axis=1);fl=pos-pos.mean(0,keepdims=True)
    cov=np.einsum('fiv,fjv->ij',fl,fl)/len(fl)
    var=np.diag(cov);corr=cov/np.sqrt(np.maximum(var[:,None]*var[None,:],1e-12));corr=np.clip(corr,-1,1)
    meanpos=pos.mean(0);tree=ag.cKDTree(meanpos);pairs=tree.query_pairs(12.0)
    g=nx.Graph();g.add_nodes_from(nodes)
    for i,j in pairs:
        strength=abs(corr[i,j])
        if strength>=0.15:g.add_edge(nodes[i],nodes[j],weight=float(-np.log(max(strength,1e-6))),corr=float(corr[i,j]))
    records=[]
    for source in ['G:34','G:35']:
        for target in ['A:832','A:925','A:1180']:
            try:
                path=nx.shortest_path(g,source,target,weight='weight');cost=nx.path_weight(g,path,'weight')
                records.append({'system':system,'source':source,'target':target,'cost':cost,'nodes':len(path),'path':' -> '.join(path)})
            except nx.NetworkXNoPath:
                records.append({'system':system,'source':source,'target':target,'cost':np.nan,'nodes':0,'path':''})
    # Direct dynamic coupling of post-break guide to the RuvC corridor.
    ni={n:i for i,n in enumerate(nodes)}
    ruv=[i for n,i in ni.items() if n.startswith('A:')]
    summary={}
    for source in ['G:34','G:35']:
        vals=np.abs(corr[ni[source],ruv]);summary[source+'_max_abs_corr_RuvC']=float(vals.max());summary[source+'_mean_abs_corr_RuvC']=float(vals.mean())
    return records,summary


def describe(x):
    a=np.asarray(x,float);blocks=np.array([z.mean() for z in np.array_split(a,8)])
    return {'mean':float(a.mean()),'sd':float(a.std()),'median':float(np.median(a)),
            'block_mean_range_2p5_97p5':[float(np.percentile(blocks,2.5)),float(np.percentile(blocks,97.5))]}


def active_water_tables(out,systems):
    rows=[];frames=[]
    for system in systems:
        path=ROOT/'precatalytic_production_100ns_3rep'/system/'rep1'/'prod_rep1_100ns_active_water.tsv'
        if not path.exists():
            continue
        d=pd.read_csv(path,sep='\t');d['system']=system;d['time_ns']=d.time_ps/1000.0
        d['mg_coord_water']=d.nearest_Mg_A<=2.6
        d['near_B18P_water']=d.O_P_A<=4.0
        d['inline_like_water']=(d.nearest_Mg_A<=3.5)&(d.O_P_A<=4.0)&(d.O_P_O3_angle_deg>=120.0)
        fr=(d.groupby(['system','step','time_ns'])
              .agg(reported_waters=('water_O_index','size'),
                   mg_coord_waters=('mg_coord_water','sum'),
                   near_B18P_waters=('near_B18P_water','sum'),
                   inline_like_waters=('inline_like_water','sum'),
                   min_O_P_A=('O_P_A','min'),
                   min_nearest_Mg_A=('nearest_Mg_A','min'),
                   max_O_P_O3_angle_deg=('O_P_O3_angle_deg','max'))
              .reset_index())
        frames.append(fr)
        rows.append({'system':system,'reported_frames':len(fr),'rows':len(d),
                     'mg_coord_water_frame_fraction':float((fr.mg_coord_waters>0).mean()),
                     'near_B18P_water_frame_fraction':float((fr.near_B18P_waters>0).mean()),
                     'inline_like_water_frame_fraction':float((fr.inline_like_waters>0).mean()),
                     'mean_min_O_P_A':float(fr.min_O_P_A.mean()),
                     'mean_max_O_P_O3_angle_deg':float(fr.max_O_P_O3_angle_deg.mean())})
    if rows:
        pd.concat(frames,ignore_index=True).to_csv(out/'active_water_frame_timeseries.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'active_water_summary.csv',index=False)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['four-system','completed-rep1'],default='four-system')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--systems',nargs='+',default=None)
    return p.parse_args()


def main():
    args=parse_args()
    outdir=args.output_dir or (ROOT/'analysis_completed_precatalytic_rep1' if args.dataset=='completed-rep1' else OUT)
    systems=args.systems or (['Match-Full','Match-Split','MM-Full'] if args.dataset=='completed-rep1' else ag.SYSTEMS)
    reference_pdb=(ROOT/'precatalytic_four_systems_staged_equilibration/Match-Full/06_unrestrained_stability_npt.pdb'
                   if args.dataset=='completed-rep1' else None)
    outdir.mkdir(parents=True,exist_ok=True);ref=global_reference(reference_pdb);dfs=[];netrows=[];netsum={}
    for system in systems:
        if args.dataset=='completed-rep1':
            tr,top,time=load_completed_rep1(system)
        else:
            tr,top,time=ag.load_system(system)
        x=align_global(tr,top,ref)
        dfs.append(feature_table(system,tr,top,x,time));nr,ns=network(system,top,x);netrows+=nr;netsum[system]=ns
        del tr,x;gc.collect()
    d=pd.concat(dfs,ignore_index=True);d.to_csv(outdir/'feature_timeseries.csv',index=False)
    pd.DataFrame(netrows).to_csv(outdir/'allosteric_network_paths.csv',index=False)
    if args.dataset=='completed-rep1':
        active_water_tables(outdir,systems)

    exclude={'system','time_ns','wet_active','DNA_nearest_P_residue','cut_nearest_B14_B24_P_residue'}
    features=[c for c in d.columns if c not in exclude]
    z=StandardScaler().fit_transform(d[features]);pca=PCA(n_components=8,random_state=0);pc=pca.fit_transform(z)
    km=KMeans(n_clusters=4,n_init=100,random_state=20260803);d['state']=km.fit_predict(pc[:,:6])
    counts=d.groupby(['system','state']).size().unstack(fill_value=0)
    occ=counts.div(counts.sum(axis=1),axis=0).reset_index();occ.to_csv(outdir/'state_occupancy.csv',index=False)
    load=pd.DataFrame(pca.components_[:4].T,index=features,columns=[f'PC{i}' for i in range(1,5)]);load.to_csv(outdir/'pca_feature_loadings.csv')
    d[['system','time_ns','wet_active','state']+[f for f in features]].to_csv(outdir/'state_assignment_timeseries.csv',index=False)
    summary={s:{c:describe(d.loc[d.system==s,c]) for c in features} for s in systems}
    # Chemistry-defined pre-catalytic proxy; water geometry is deliberately absent.
    ready=(d.Mg_Mg_A.between(3,5.5)&(d[['Mg1_D832_A','Mg1_E925_A','Mg2_D832_A','Mg2_D1180_A']]<2.6).all(axis=1)
           &(d.B18_OP_to_Mg_min_A<3.0))
    ready_frac={s:float(ready[d.system==s].mean()) for s in systems}
    result={'wet_activity_labels':WET_ACTIVE,'dataset':args.dataset,'systems':systems,'features':features,'pca_explained_variance':pca.explained_variance_ratio_.tolist(),
         'precatalytic_proxy_definition':'Mg-Mg 3-5.5 A; four canonical metal-carboxylate minima <2.6 A; B18 phosphate O-Mg <3.0 A; attacking water unavailable',
         'precatalytic_proxy_fraction':ready_frac,'network_summary':netsum,'summary':summary,
         'limitations':['one trajectory per condition','active-water reporter summarized separately when available','completed-rep1 mode is a partial set until remaining replicas finish']}
    (outdir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

    fig,ax=plt.subplots(1,2,figsize=(12,5))
    colors={'Match-Full':'#1f77b4','Match-Split':'#2ca02c','MM-Full':'#ff7f0e','MM-Split':'#d62728'}
    for s in ag.SYSTEMS:
        if s not in systems: continue
        m=d.system==s;ax[0].scatter(pc[m,0],pc[m,1],s=20,alpha=.65,label=s,color=colors[s]);ax[1].plot(d.loc[m,'time_ns'],d.loc[m,'state'],'.-',ms=3,lw=.5,label=s,color=colors[s])
    ax[0].set(xlabel='PC1',ylabel='PC2',title='Joint pre-catalytic/allosteric feature space');ax[0].legend(fontsize=8)
    ax[1].set(xlabel='Time (ns)',ylabel='State',title='Four-state assignment');ax[1].set_yticks(range(4));ax[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(outdir/'state_space_and_occupancy.png',dpi=200);plt.close(fig)
    print(json.dumps({'rows':len(d),'features':len(features),'ready_fraction':ready_frac,'output':str(outdir)},indent=2))


if __name__=='__main__':main()

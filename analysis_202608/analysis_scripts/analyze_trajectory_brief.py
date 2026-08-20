from __future__ import annotations

import argparse, json, gc
from pathlib import Path
import mdtraj as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/'production_md_2p5mM_MgCl2_37p5mM_KCl'
EQ=ROOT/'equilibration_smoke_2p5mM_MgCl2_37p5mM_KCl'
DOMAINS={'RuvC-I':(809,872),'RuvC-II':(891,997),'RuvC-III':(1180,1226)}
SYSTEMS=['Match-Full','Match-Split','MM-Full','MM-Split']

def st(x):
 x=np.asarray(x,float); return {'mean':float(x.mean()),'sd':float(x.std()),'p05':float(np.percentile(x,5)),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'min':float(x.min()),'max':float(x.max())}
def atom(top,ch,res,name): return next(a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res and a.name==name)
def selres(top,ch,lo,hi,heavy=True): return np.array([a.index for a in top.atoms if a.residue.chain.chain_id==ch and lo<=a.residue.resSeq<=hi and (not heavy or (a.element and a.element.symbol!='H'))],int)
def baseidx(top,ch,res):
 sugar={"P","OP1","OP2","OP3","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
 return np.array([a.index for a in top.atoms if a.residue.chain.chain_id==ch and a.residue.resSeq==res and a.name not in sugar and a.element and a.element.symbol!='H'],int)
def repair(top):
 for c in top.chains:
  rs=list(c.residues)
  if rs and rs[0].is_nucleic:
   for l,r in zip(rs[:-1],rs[1:]):
    a=next((x for x in l.atoms if x.name in {"O3'","O3*"}),None); b=next((x for x in r.atoms if x.name=='P'),None)
    if a and b: top.add_bond(a,b)
def dist(t,i,j): return md.compute_distances(t,[[int(i),int(j)]],periodic=True)[:,0]*10
def rmsd_internal(t,idx):
 q=t[:]; q.superpose(q,0,atom_indices=idx); return md.rmsd(q,q,0,atom_indices=idx)*10
def raw_rmsd(t,idx): return np.sqrt(np.mean(np.sum((t.xyz[:,idx]-t.xyz[0,idx])**2,axis=2),axis=1))*10
def contact_count(t,a,b,cut=.45):
 pairs=np.array([(i,j) for i in a for j in b],dtype=int); out=np.zeros(t.n_frames)
 for k in range(0,len(pairs),200000): out+=np.sum(md.compute_distances(t,pairs[k:k+200000],periodic=True)<cut,axis=1)
 return out

def analyze(system,warmup_ns=20.):
 pref=PROD/system/'prod_0001_100ns'; out=PROD/system/'analysis_brief'; out.mkdir(exist_ok=True)
 if system=='MM-Split':
  prefixes=[PROD/system/f'prod_{i:04d}' for i in range(1,6)]+[PROD/system/'prod_0006_95ns']
 else:
  prefixes=[pref]
 idx=np.loadtxt(str(prefixes[0])+'_solute_atom_indices.txt',dtype=int)
 full=md.load_pdb(str(EQ/system/'06_unrestrained_stability_npt.pdb')); top=full.atom_slice(idx).topology; del full; repair(top)
 chunks=[md.load(str(p)+'_solute.dcd',top=top) for p in prefixes]
 if system=='MM-Split':
  # The five 1 ns segments were saved every 50 ps, while the 95 ns segment
  # was saved every 100 ps.  Retain 100, 200, ... ps from the short segments
  # so all four systems use the same 100 ps sampling cadence.
  chunks=[chunk[1::2] for chunk in chunks[:5]]+[chunks[5]]
 tr=md.join(chunks,check_topology=True,discard_overlapping_frames=False); del chunks
 assert tr.n_atoms==len(idx) and tr.n_frames==1000
 mol=list(top.find_molecules()); anchor=max(mol,key=len); tr=tr.image_molecules(anchor_molecules=[anchor],other_molecules=[m for m in mol if m is not anchor],make_whole=True)
 start=int(round(warmup_ns/.1)); tr=tr[start:]; time=np.arange(start,start+tr.n_frames)*.1
 core=np.array([a.index for a in top.atoms if a.residue.chain.chain_id=='A' and 1<=a.residue.resSeq<=1230 and a.name in {'N','CA','C'}],int)
 tr.superpose(tr,0,atom_indices=core)
 ca=np.array([a.index for a in top.atoms if a.residue.chain.chain_id=='A' and a.name=='CA'],int)
 dna=selres(top,'B',1,999); rna_ch=['C']+(['D'] if 'Split' in system else []); rna=np.concatenate([selres(top,c,1,999) for c in rna_ch])
 metrics={'time_ns':time,'protein_rmsd_A':md.rmsd(tr,tr,0,atom_indices=core)*10,'dna_rmsd_A':rmsd_internal(tr,dna),'rna_rmsd_A':rmsd_internal(tr,rna)}
 metrics['protein_rg_A']=md.compute_rg(tr.atom_slice(selres(top,'A',1,1249)))*10; metrics['dna_rg_A']=md.compute_rg(tr.atom_slice(dna))*10; metrics['rna_rg_A']=md.compute_rg(tr.atom_slice(rna))*10
 domains={}; domain_series={}
 for name,(lo,hi) in DOMAINS.items():
  ids=selres(top,'A',lo,hi); x=raw_rmsd(tr,ids); domain_series[name]=x; metrics[name+'_rmsd_A']=x; domains[name]={'global_frame_rmsd_A':st(x),'internal_rmsd_A':st(rmsd_internal(tr,ids))}
 allruv=np.concatenate([selres(top,'A',*r) for r in DOMAINS.values()]); x=raw_rmsd(tr,allruv); metrics['RuvC-overall_rmsd_A']=x; domains['RuvC-overall']={'global_frame_rmsd_A':st(x),'internal_rmsd_A':st(rmsd_internal(tr,allruv))}
 mg=np.array([a.index for a in top.atoms if a.element and a.element.symbol=='Mg']); assert len(mg)==2
 pairs={}
 for mn,mi in [('Mg1',mg[0]),('Mg2',mg[1])]:
  for res,names in [(832,['OD1','OD2']),(925,['OE1','OE2']),(1180,['OD1','OD2'])]:
   for name in names: pairs[f'{mn}_{res}_{name}']=(mi,atom(top,'A',res,name))
 if system=='Match-Full':
  pairs.update({'Mg1_B18_OP1':(mg[0],atom(top,'B',18,'OP1')),'Mg2_B18_OP2':(mg[1],atom(top,'B',18,'OP2'))})
 ds={k:dist(tr,*v) for k,v in pairs.items()}; mgmg=dist(tr,*mg); metrics['Mg_Mg_A']=mgmg
 for k,v in ds.items(): metrics[k+'_A']=v
 canonical={'Mg1_D832_min':np.minimum(ds['Mg1_832_OD1'],ds['Mg1_832_OD2']),'Mg1_E925_min':np.minimum(ds['Mg1_925_OE1'],ds['Mg1_925_OE2']),'Mg2_D832_min':np.minimum(ds['Mg2_832_OD1'],ds['Mg2_832_OD2']),'Mg2_D1180_min':np.minimum(ds['Mg2_1180_OD1'],ds['Mg2_1180_OD2'])}
 productive=(mgmg>=3)&(mgmg<=5.5)
 for v in canonical.values(): productive&=v<2.6
 pocket=selres(top,'A',832,832); pocket=np.concatenate([pocket,selres(top,'A',925,925),selres(top,'A',1180,1180),mg]); pr=md.rmsd(tr,tr,0,atom_indices=pocket)*10; metrics['pocket_rmsd_A']=pr
 # M17: B26 pairs with full C38 or split D4.
 mch,mres=('D',4) if 'Split' in system else ('C',38); b26=baseidx(top,'B',26); mr=baseidx(top,mch,mres)
 bpairs=np.array([(i,j) for i in b26 for j in mr],int); bd=md.compute_distances(tr,bpairs,periodic=True)*10
 metrics['M17_min_base_distance_A']=bd.min(1); metrics['M17_polar_contacts_lt3p5A']=np.sum(bd<3.5,axis=1)
 mlocal=np.concatenate([selres(top,'B',23,29),selres(top,mch,max(1,mres-3),mres+3)]); metrics['M17_local_rmsd_A']=md.rmsd(tr,tr,0,atom_indices=mlocal)*10
 # RuvC-II/crRNA backbone contacts and relative position.
 r2=selres(top,'A',891,997); rb=np.array([a.index for a in top.atoms if a.residue.chain.chain_id in rna_ch and a.name in {'P',"O5'","O3'"}],int)
 metrics['RuvCII_crRNA_contacts_lt4p5A']=contact_count(tr,r2,rb)
 r2ca=np.array([a.index for a in top.atoms if a.residue.chain.chain_id=='A' and 891<=a.residue.resSeq<=997 and a.name=='CA']); r2com=tr.xyz[:,r2ca].mean(1); mgcom=tr.xyz[:,mg].mean(1); metrics['RuvCII_to_Mg_centroid_A']=np.linalg.norm(r2com-mgcom,axis=1)*10
 split={}
 if 'Split' in system:
  br=dist(tr,atom(top,'C',34,"O3'"),atom(top,'D',1,"O5'")); metrics['split_break_O3_O5_A']=br
  front=selres(top,'C',31,34); back=selres(top,'D',1,4); metrics['split_local_contacts_lt4p5A']=contact_count(tr,front,back)
  dfrag=selres(top,'D',1,10); metrics['split_back_fragment_positional_rmsd_A']=raw_rmsd(tr,dfrag)
  split={k:st(metrics[k]) for k in ['split_break_O3_O5_A','split_local_contacts_lt4p5A','split_back_fragment_positional_rmsd_A']}
 # Thermodynamics, same 20-100 ns window.
 state_chunks=[np.atleast_2d(np.loadtxt(str(p)+'.tsv',skiprows=1,usecols=range(9))) for p in prefixes]
 if system=='MM-Split': state_chunks=[chunk[1::2] for chunk in state_chunks[:5]]+[state_chunks[5]]
 state=np.concatenate(state_chunks,axis=0)[start:]
 thermo={'temperature_K':st(state[:,5]),'density_g_ml':st(state[:,7]),'potential_energy_kj_mol':st(state[:,2]),'total_energy_kj_mol':st(state[:,4])}
 for k,c in [('potential_slope_kj_mol_ns',2),('total_slope_kj_mol_ns',4),('temperature_slope_K_ns',5),('density_slope_g_ml_ns',7)]: thermo[k]=float(np.polyfit(time,state[:,c],1)[0])
 # Catalytic/scissile SASA (standard solute SASA; solvent coordinates not required).
 sasa=md.shrake_rupley(tr,mode='atom',n_sphere_points=160)*100
 sas={}
 for res in [832,925,1180]: sas[f'A:{res}']=st(sasa[:,selres(top,'A',res,res)].sum(1))
 ph=np.array([atom(top,'B',18,n) for n in ['P','OP1','OP2']]); sas['B:18_scissile_phosphate']=st(sasa[:,ph].sum(1))
 # Residue/nucleotide RMSF after global alignment.
 rows=[]; meanxyz=tr.xyz.mean(0); atom_rmsf=np.sqrt(np.mean(np.sum((tr.xyz-meanxyz[None,:,:])**2,axis=2),axis=0))*10
 for c in ['A','B']+rna_ch:
  for res in [r for r in top.residues if r.chain.chain_id==c]:
   ids=np.array([a.index for a in res.atoms if a.element and a.element.symbol!='H']);
   if len(ids): rows.append({'chain':c,'resSeq':res.resSeq,'resname':res.name,'rmsf_A':float(atom_rmsf[ids].mean())})
 pd.DataFrame(rows).to_csv(out/'rmsf_by_residue.csv',index=False)
 df=pd.DataFrame(metrics); df.to_csv(out/'timeseries_20_100ns.csv',index=False)
 # Clustering on protein CA + nucleic phosphates.
 feat=np.concatenate([ca,np.array([a.index for a in top.atoms if a.residue.chain.chain_id in ['B']+rna_ch and a.name=='P'])]); Z=PCA(10,random_state=0).fit_transform(tr.xyz[:,feat].reshape(tr.n_frames,-1)); labels=KMeans(3,n_init=20,random_state=0).fit_predict(Z); clusters=[]
 for k in range(3):
  m=np.where(labels==k)[0]; center=Z[m].mean(0); rep=int(m[np.argmin(np.sum((Z[m]-center)**2,axis=1))]); tr[rep].save_pdb(str(out/f'cluster_{k+1}_representative_{time[rep]:.1f}ns.pdb')); clusters.append({'cluster':k+1,'fraction':len(m)/len(labels),'representative_ns':float(time[rep])})
 corr={}
 keys=['M17_min_base_distance_A','M17_polar_contacts_lt3p5A','RuvC-II_rmsd_A','RuvCII_crRNA_contacts_lt4p5A','RuvCII_to_Mg_centroid_A','pocket_rmsd_A','Mg_Mg_A']
 if 'Split' in system: keys+=['split_break_O3_O5_A','split_back_fragment_positional_rmsd_A']
 for i,a in enumerate(keys):
  for b in keys[i+1:]: corr[a+'__'+b]=float(np.corrcoef(metrics[a],metrics[b])[0,1])
 summary={'system':system,'topology_atoms':len(idx),'trajectory_frames':1000,'frame_stride_ps':100,'warmup_discarded_ns':20,'analyzed_window_ns':'20-100','replicas':1,'forcefield':['amber14-all.xml','amber14/tip3p.xml'],'water_model':'TIP3P','temperature_K':310.15,'pressure_atm_target':1.0,'salt':'2.5 mM MgCl2 + 37.5 mM KCl','alignment':'chain A residues 1-1230 backbone N,CA,C','global':{k:st(metrics[k]) for k in ['protein_rmsd_A','dna_rmsd_A','rna_rmsd_A','protein_rg_A','dna_rg_A','rna_rg_A']},'thermodynamics':thermo,'RuvC_domains':domains,'catalytic':{'Mg_Mg_A':st(mgmg),'distances_A':{k:st(v) for k,v in ds.items()},'canonical_min_distances_A':{k:st(v) for k,v in canonical.items()},'scissile_phosphate_assignment':('B:18, directly coordinated in Match-Full' if system=='Match-Full' else 'not assigned: no DNA phosphate within the catalytic Mg first shell'),'pocket_rmsd_A':st(pr),'productive_definition':'Mg-Mg 3-5.5 A and canonical minimum metal-sidechain oxygen distances <2.6 A; scissile phosphate reported separately','productive_fraction':float(productive.mean())},'M17':{'DNA':'B:26','guide':f'{mch}:{mres}','min_base_distance_A':st(metrics['M17_min_base_distance_A']),'polar_contact_count':st(metrics['M17_polar_contacts_lt3p5A']),'local_rmsd_A':st(metrics['M17_local_rmsd_A'])},'split':split,'SASA_A2':sas,'correlations':corr,'clusters':clusters,'limitations':['single replicate per condition','solute-only trajectory: no catalytic-water, water-mediated H-bond, or ion-distribution analysis','instantaneous pressure not recorded']}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,ax=plt.subplots(3,2,figsize=(12,10),sharex=True); plotkeys=['protein_rmsd_A','rna_rmsd_A','dna_rmsd_A','RuvC-II_rmsd_A','Mg_Mg_A','M17_min_base_distance_A']
 for a,k in zip(ax.flat,plotkeys): a.plot(time,metrics[k],lw=.7); a.set_ylabel(k); a.grid(alpha=.2)
 ax[-1,0].set_xlabel('Time (ns)'); ax[-1,1].set_xlabel('Time (ns)'); fig.tight_layout(); fig.savefig(out/'core_metrics_timeseries.png',dpi=180); plt.close(fig)
 del tr,sasa; gc.collect(); return summary

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--system',choices=SYSTEMS); ap.add_argument('--warmup-ns',type=float,default=20); a=ap.parse_args(); systems=[a.system] if a.system else SYSTEMS
 summaries=[analyze(s,a.warmup_ns) for s in systems]; print(json.dumps([{s:{'productive':x['catalytic']['productive_fraction'],'protein_rmsd':x['global']['protein_rmsd_A']['mean']}} for s,x in zip(systems,summaries)],indent=2))
if __name__=='__main__': main()

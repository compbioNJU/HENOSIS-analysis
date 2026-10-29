#!/usr/bin/env python3
"""Recover cell-level violin input and validate saved donor means; inputs read-only."""
from pathlib import Path
import os
import h5py,numpy as np,pandas as pd
from scipy import sparse
PAPER=Path(__file__).resolve().parents[4]
BRAIN=Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/aging'
OUT=Path(os.environ.get('FIGURE4_OUTPUT_DIR',PAPER/'review_output/Figure4'))/'results/attention_aging/source_data/Fig4j';OUT.mkdir(parents=True,exist_ok=True);DATA=Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/attention_aging'

def read(group,key):
 x=group[key]
 if isinstance(x,h5py.Group):
  cats=x['categories'].asstr()[:];codes=x['codes'][:];return np.array([cats[c] if c>=0 else 'nan' for c in codes],object)
 if x.dtype.kind in 'OSU':return x.asstr()[:]
 return x[:]

for species,genes in [('human',['OLIG2']),('mouse',['Olig2'])]:
 path=BRAIN/f'{species}_concated_adata_with_meta.h5ad'
 with h5py.File(path,'r') as f:
  obs=f['obs'];stage=read(obs,'age_stage');ct=read(obs,'final_annotation');sample=read(obs,'sample_id' if species=='human' else 'mouse_id');ids=read(obs,'_index');donor=read(obs,'donor_id') if species=='human' else sample;var=read(f['var'],'_index');inds=np.flatnonzero((ct=='OPC')&np.isin(stage,['young','middle','old']))
  print(species,'OPC cells',len(inds),flush=True)
  idx={g:next(i for i,v in enumerate(var) if str(v).upper()==g.upper()) for g in genes}
  x=f['X']
  if isinstance(x,h5py.Group):
   csr=sparse.csr_matrix((x['data'][:],x['indices'][:],x['indptr'][:]),shape=tuple(x.attrs['shape']))
   values={g:np.asarray(csr[inds,j].toarray()).ravel() for g,j in idx.items()}
  else:
   # Read contiguous row chunks from the dense matrix, selecting only target genes.
   values={g:np.empty(len(inds),dtype=np.float64) for g in genes}
   for start in range(0,len(stage),20000):
    end=min(start+20000,len(stage));a=np.searchsorted(inds,start);b=np.searchsorted(inds,end)
    if b==a:continue
    for g,j in idx.items():values[g][a:b]=x[start:end,j][inds[a:b]-start]
  for gene in genes:
   raw=values[gene].astype(float); log_applied=raw.max()>20;expr=np.log1p(raw) if log_applied else raw
   df=pd.DataFrame({'cell_id':ids[inds].astype(str),'sample_id':sample[inds].astype(str),'donor_id':donor[inds].astype(str),'age_stage':stage[inds].astype(str),'gene':gene,'expression_used_in_plot':expr,'raw_X':raw})
   df[['sample_id','donor_id','age_stage']].drop_duplicates().sort_values(['age_stage','sample_id']).to_csv(OUT/f'{species}_{gene}_sample_to_donor.csv',index=False)
   df.to_csv(OUT/f'{species}_{gene}_OPC_all_cells.csv.gz',index=False,compression='gzip')
   sampled=df.groupby('age_stage',observed=True,group_keys=False).apply(lambda g:g.sample(n=min(len(g),3000),random_state=0)).reset_index(drop=True)
   sampled.to_csv(OUT/f'{species}_{gene}_OPC_violin_sampled_cells.csv',index=False)
   summ=df.groupby(['sample_id','age_stage'],observed=True).agg(n_cells=('expression_used_in_plot','size'),mean_expr_log=('expression_used_in_plot','mean'),median_expr_log=('expression_used_in_plot','median')).reset_index()
   original=pd.read_csv(DATA/f'{species}_OPC_all_gene_donor_summaries.csv',dtype={'mouse_id':str,'sample_id':str});original=original[original.gene.eq(gene)].rename(columns={'mouse_id':'sample_id'})
   merged=summ.merge(original,on=['sample_id','age_stage'],validate='one_to_one',suffixes=('_recovered','_original'))
   assert len(merged)==len(original)==len(summ)
   for column in ['n_cells','mean_expr_log','median_expr_log']:
    np.testing.assert_allclose(merged[column+'_recovered'],merged[column+'_original'],rtol=1e-6,atol=1e-7)
   summ.to_csv(OUT/f'{species}_{gene}_OPC_sample_summary.csv',index=False)
   print(species,gene,'log1p',log_applied,'cells',len(df),'sampled',len(sampled),'samples',len(summ),'summary verified',flush=True)

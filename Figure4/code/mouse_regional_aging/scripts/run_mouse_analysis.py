"""Mouse-only leave-one-animal-out age occupancy and full pipeline permutations."""
import argparse,json,time
import numpy as np,pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from settings import ROOT,CFG,load,RESULTS
REGIONS=CFG['regions'];GLIA=CFG['glial_lineages'];STAGES=['young','middle','old']
def axis(centers,stage):
 y=stage==0;o=stage==2
 a=(centers[o].sum(0)-o[:,None]*centers)/(o.sum()-o[:,None]);b=(centers[y].sum(0)-y[:,None]*centers)/(y.sum()-y[:,None]);v=a-b
 return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
def occupancy(data,axes,stage,q=CFG['reference_quantile']):
 quant=[];own=[]
 for i,x in enumerate(data):
  if len(x)<CFG['minimum_cells_per_mouse_region']:raise ValueError('Not enough cells in a mouse/reference region')
  p=x@axes.T;quant.append(np.quantile(p,q,axis=0));own.append(p[:,i])
 quant=np.asarray(quant);y=stage==0;threshold=(quant[y].sum(0)-np.diag(quant)*y)/(y.sum()-y)
 return np.asarray([np.mean(v>threshold[i]) for i,v in enumerate(own)]),threshold
def rankcorr(a,b):
 a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean();return float(a@b/np.sqrt((a@a)*(b@b))) if b@b else 0.
def evaluate(data,stage,q=CFG['reference_quantile']):
 a=axis(data['centers'],stage);occ,threshold=occupancy(data['lineage'],a,stage,q)
 regional={};t={};metrics={'delta':float(occ[stage==2].mean()-occ[stage==0].mean())}
 for region,x in data['regional'].items():regional[region],t[region]=occupancy(x,a,stage,q)
 if regional:metrics['regional_rho']=rankcorr(stage,regional['CC/ACO']-regional['CTX'])
 return occ,threshold,regional,t,metrics
def prepare(ct):
 meta,X,_,_=load('Mouse',ct);donors=sorted(meta.donor.unique());rng=np.random.default_rng(CFG['seed']);full=[];small=[];regions={r:[] for r in REGIONS} if ct in GLIA else {};sample_regions={r:[] for r in regions};stage=[]
 for donor in donors:
  g=meta[meta.donor.eq(donor)];assert g.age_stage.nunique()==1
  stage.append(STAGES.index(g.age_stage.iloc[0]));v=X[g.index.to_numpy()];full.append(v);small.append(v[rng.choice(len(v),min(CFG['fixed_lineage_sample_cap'],len(v)),replace=False)])
  for r in regions:
   x=X[g[g.region.eq(r)].index.to_numpy()];regions[r].append(x);sample_regions[r].append(x[rng.choice(len(x),min(CFG['fixed_region_sample_cap'],len(x)),replace=False)])
 return donors,np.array(stage),dict(lineage=full,centers=np.array([v.mean(0) for v in full]),regional=regions),dict(lineage=small,centers=np.array([v.mean(0) for v in small]),regional=sample_regions)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--permutations',type=int,default=1000);args=parser.parse_args();rng=np.random.default_rng(CFG['seed']);datasets={};rows=[];regrows=[];effects=[];observed={}
 for ct in CFG['lineages']:
  donors,stage,full,small=prepare(ct);datasets[ct]=(donors,stage,full,small);occ,thr,reg,rt,metrics=evaluate(full,stage);sm=evaluate(small,stage)[-1]
  assert len(donors)==20 and list(np.bincount(stage))==[4,7,9]
  for i,d in enumerate(donors):
   rows.append(dict(cell_type=ct,donor=d,age_stage=STAGES[stage[i]],stage=stage[i],n_cells=len(full['lineage'][i]),occupancy=occ[i],occupancy_pct=100*occ[i],training_threshold=thr[i]))
   for r in reg:regrows.append(dict(cell_type=ct,donor=d,age_stage=STAGES[stage[i]],stage=stage[i],region=r,n_cells=len(full['regional'][r][i]),occupancy=reg[r][i],occupancy_pct=100*reg[r][i],training_threshold=rt[r][i],CC_minus_CTX_pp=100*(reg['CC/ACO'][i]-reg['CTX'][i])))
  e=dict(cell_type=ct,young_mean_pct=100*occ[stage==0].mean(),middle_mean_pct=100*occ[stage==1].mean(),old_mean_pct=100*occ[stage==2].mean(),delta_pp=100*metrics['delta'],regional_rho=metrics.get('regional_rho',np.nan));effects.append(e)
  for key,v in sm.items():observed[ct+'|'+key]=v
  print(ct,e,flush=True)
 for name,data in [('individual_occupancy',rows),('individual_region_occupancy',regrows),('full_cell_effects',effects)]:pd.DataFrame(data).to_csv(RESULTS/f'tables/{name}.csv',index=False)
 null=[];stage0=datasets[CFG['lineages'][0]][1];t0=time.time()
 for j in range(args.permutations):
  stage=rng.permutation(stage0);row={'replicate':j}
  for ct,(donors,original,full,small) in datasets.items():
   assert np.array_equal(original,stage0)
   for key,v in evaluate(small,stage)[-1].items():row[ct+'|'+key]=v
  null.append(row)
  if (j+1)%100==0:print('permutations',j+1,'seconds',round(time.time()-t0,1),flush=True)
 null=pd.DataFrame(null);null.to_csv(RESULTS/'tables/full_pipeline_null.csv.gz',index=False)
 tests=[]
 for name,v in observed.items():
  ct,metric=name.split('|');p=(1+np.sum(null[name]>=v))/(args.permutations+1);tests.append(dict(cell_type=ct,metric=metric,observed_subsample=v,p_one_sided=p,n_permutations=args.permutations))
 tests=pd.DataFrame(tests);tests.to_csv(RESULTS/'tables/full_pipeline_permutation_tests.csv',index=False)
 print(tests.to_string(index=False),flush=True)
if __name__=='__main__':main()

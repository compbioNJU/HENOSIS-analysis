"""Independently recompute all two-sided mouse tests, sharing identical label draws."""
import json,time,multiprocessing,concurrent.futures
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
from settings import ROOT,CFG,CACHE,RESULTS
from run_mouse_analysis import prepare,evaluate
OUT=RESULTS/'tables'
def calculate_metrics(ds,stage):
 _,_,reg,_,m=evaluate(ds,stage)
 if reg:
  contrast=reg['CC/ACO']-reg['CTX'];m['region_interaction']=float(contrast[stage==2].mean()-contrast[stage==0].mean())
 return m

def worker(ct,small,labels):
 t=time.time();values=[]
 for j,stage in enumerate(labels):
  values.append(calculate_metrics(small,stage))
  if (j+1)%2000==0:print(ct,'permutations',j+1,'seconds',round(time.time()-t),flush=True)
 return ct,pd.DataFrame(values)

def main():
 data={};observed={};full={};stages=[]
 for ct in CFG['lineages']:
  donors,stage,allcells,small=prepare(ct);data[ct]=small;stages.append(stage)
  full.update({ct+'|'+k:v for k,v in calculate_metrics(allcells,stage).items()});observed.update({ct+'|'+k:v for k,v in calculate_metrics(small,stage).items()})
 assert all(np.array_equal(stages[0],s) for s in stages)
 rng=np.random.default_rng(CFG['seed']);labels=np.array([rng.permutation(stages[0]) for _ in range(CFG['permutations'])]);np.save(CACHE/'permuted_mouse_age_labels.npy',labels)
 order=['Oligodendrocyte','Microglia']+[c for c in CFG['lineages'] if c not in ['Oligodendrocyte','Microglia']];results={}
 with concurrent.futures.ProcessPoolExecutor(max_workers=CFG['permutation_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
  tasks=[pool.submit(worker,c,data[c],labels) for c in order]
  for future in concurrent.futures.as_completed(tasks):
   ct,table=future.result();results[ct]=table;print('Completed',ct,flush=True)
 null=pd.DataFrame({'replicate':np.arange(len(labels))})
 for ct in CFG['lineages']:
  for name in results[ct]:null[ct+'|'+name]=results[ct][name].to_numpy()
 null.to_csv(OUT/'mouse_full_pipeline_null_9999.csv.gz',index=False);null.to_parquet(OUT/'mouse_full_pipeline_null_9999.parquet',index=False)
 records=[]
 for name,value in observed.items():
  ct,metric=name.split('|');v=null[name].to_numpy();upper=(1+sum(v>=value))/(len(v)+1);lower=(1+sum(v<=value))/(len(v)+1)
  records.append(dict(cell_type=ct,metric=metric,full_cell_effect=full[name],subsample_observed=value,p_upper=upper,p_lower=lower,p_two=min(1.,2*min(upper,lower)),null_mean=v.mean(),null_median=np.median(v),null_q025=np.quantile(v,.025),null_q975=np.quantile(v,.975),permutations=len(v)))
 tests=pd.DataFrame(records);interactions=tests.metric.eq('region_interaction');assert interactions.sum()==2;tests['q_BH_interactions_two']=np.nan;tests.loc[interactions,'q_BH_interactions_two']=multipletests(tests.loc[interactions,'p_two'],method='fdr_bh')[1];tests.to_csv(OUT/'mouse_full_pipeline_twosided.csv',index=False)
 old=pd.read_csv(OUT/'full_pipeline_null.csv.gz');cols=[c for c in old if c!='replicate'];np.testing.assert_allclose(old[cols],null.loc[:999,cols],atol=1e-12)
 (OUT/'permutation_reproducibility.json').write_text(json.dumps({'status':'passed','n_permutations':len(null),'independently_recalculated_here':True,'first_1000_same_as_local_baseline':True,'shared_age_labels_across_lineages':True,'exploratory_interaction_tests':int(interactions.sum())},indent=2)+'\n');print(tests.to_string(index=False),flush=True)
if __name__=='__main__':main()

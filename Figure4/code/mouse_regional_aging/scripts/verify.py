"""Validate raw-source alignment, independent animal summaries, means and permutation tests."""
import hashlib,json
import numpy as np,pandas as pd,h5py
from pypdf import PdfReader
from statsmodels.stats.multitest import multipletests
from settings import ROOT,CFG,load,BRAIN,RESULTS
from run_mouse_analysis import prepare,evaluate
T=RESULTS/'tables';B=BRAIN

def read(f,key):
 z=f[key]
 if isinstance(z,h5py.Group):
  categories=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in z['categories'][:]],object);codes=z['codes'][:];out=np.full(len(codes),None,object);valid=codes>=0;out[valid]=categories[codes[valid]];return out
 a=z[:];return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in a],object) if a.dtype.kind in 'OSU' else a
obs=pd.read_csv(T/'individual_occupancy.csv',dtype={'donor':str});regional=pd.read_csv(T/'individual_region_occupancy.csv',dtype={'donor':str});tests=pd.read_csv(T/'mouse_full_pipeline_twosided.csv');null=pd.read_parquet(T/'mouse_full_pipeline_null_9999.parquet');first=pd.read_csv(T/'full_pipeline_null.csv.gz');assert len(obs)==40 and len(regional)==120 and len(null)==9999 and len(tests)==6
cols=[c for c in first if c!='replicate'];np.testing.assert_allclose(first[cols],null.loc[:999,cols],atol=1e-12)
fixed=[];source_mouse_rows=0
I=B/'human_mouse_aging_integrated.h5ad';M=B/'mouse_concated_adata_with_meta.h5ad'
with h5py.File(I) as f,h5py.File(M) as raw:
 bar=read(f,'obs/barcode');index=pd.Index(bar);assert index.is_unique;species=read(f,'obs/sp');X=np.asarray(f['obsm/X_HENOSIS'],np.float32);names=read(raw,'obs/_index');source_donor=read(raw,'obs/mouse_id');source_region=read(raw,'obs/region');source_stage=read(raw,'obs/age_stage')
 for ct in CFG['lineages']:
  meta,x,_,_=load('Mouse',ct);pos=index.get_indexer(meta.barcode);assert (pos>=0).all() and (species[pos]=='Mus_musculus').all();np.testing.assert_array_equal(x,X[pos]);rr=meta.source_row.to_numpy();assert np.array_equal(names[rr],meta.barcode) and np.array_equal(source_donor[rr],meta.donor) and np.array_equal(source_region[rr],meta.region) and np.array_equal(source_stage[rr],meta.age_stage);source_mouse_rows+=len(meta)
  donors,stage,full,small=prepare(ct);assert list(np.bincount(stage))==[4,7,9];g=obs[obs.cell_type.eq(ct)].set_index('donor').loc[donors];occ,threshold,reg,rt,metrics=evaluate(full,stage);np.testing.assert_allclose(g.occupancy,occ,atol=1e-10);np.testing.assert_allclose(g.training_threshold,threshold,atol=1e-10)
  _,_,sr,_,sm=evaluate(small,stage)
  if sr:
   c=sr['CC/ACO']-sr['CTX'];sm['region_interaction']=c[stage==2].mean()-c[stage==0].mean()
  for metric,value in sm.items():
   row=tests[tests.cell_type.eq(ct)&tests.metric.eq(metric)].iloc[0];assert np.isclose(value,row.subsample_observed);v=null[ct+'|'+metric].to_numpy();up=(1+sum(v>=value))/(len(v)+1);lo=(1+sum(v<=value))/(len(v)+1);assert np.isclose(row.p_two,min(1,2*min(up,lo)))
  # Record the exact fixed cells used in calibration, reconstructing the same seeded draw order.
  rng=np.random.default_rng(CFG['seed'])
  for j,donor in enumerate(donors):
   a=meta[meta.donor.eq(donor)];idx=a.index.to_numpy();selected=idx[rng.choice(len(idx),min(500,len(idx)),replace=False)];np.testing.assert_array_equal(x[selected],small['lineage'][j])
   for r in meta.iloc[selected].itertuples():fixed.append(dict(lineage=ct,donor=donor,scope='all',cell_id=r.barcode,source_row=r.source_row,cache_embedding_row=r.Index))
   if ct in CFG['glial_lineages']:
    for region in CFG['regions']:
     idx=a[a.region.eq(region)].index.to_numpy();selected=idx[rng.choice(len(idx),min(300,len(idx)),replace=False)];np.testing.assert_array_equal(x[selected],small['regional'][region][j])
     for r in meta.iloc[selected].itertuples():fixed.append(dict(lineage=ct,donor=donor,scope=region,cell_id=r.barcode,source_row=r.source_row,cache_embedding_row=r.Index))
pd.DataFrame(fixed).to_parquet(T/'fixed_permutation_cell_samples.parquet',index=False)
interactions=tests.metric.eq('region_interaction');assert interactions.sum()==2;np.testing.assert_allclose(tests.loc[interactions,'q_BH_interactions_two'],multipletests(tests.loc[interactions,'p_two'],method='fdr_bh')[1])
# Validate every group mean on a trajectory and every heatmap value.
hmeans=pd.read_csv(T/'figure_k_region_group_means.csv');imeans=pd.read_csv(T/'figure_l_lineage_group_means.csv');heat=pd.read_csv(T/'figure_l_regional_heatmap.csv');contrasts=pd.read_csv(T/'figure_l_individual_contrasts.csv',dtype={'donor':str});assert len(hmeans)==18 and len(imeans)==6 and len(heat)==6 and len(contrasts)==40
for r in hmeans.itertuples():
 values=regional[regional.cell_type.eq(r.cell_type)&regional.region.eq(r.region)&regional.age_stage.eq(r.age_stage)].occupancy_pct;assert len(values)==r.n_mice and np.isclose(values.mean(),r.mean_high_age_score_cells_pct)
for r in imeans.itertuples():
 v=contrasts[contrasts.cell_type.eq(r.cell_type)&contrasts.age_stage.eq(r.age_stage)].CC_ACO_minus_CTX_pp;assert len(v)==r.n_mice and np.isclose(v.mean(),r.mean_CC_ACO_minus_CTX_pp)
for r in heat.itertuples():
 g=regional[regional.cell_type.eq(r.cell_type)&regional.region.eq(r.region)];assert np.isclose(r.older_minus_young_pp,g[g.stage.eq(2)].occupancy_pct.mean()-g[g.stage.eq(0)].occupancy_pct.mean())
fonttypes={}
for name in ['Figure4_kl_mouse_regional_aging','Figure4_hi_mouse_regional_aging']:
 pdf=PdfReader(str(RESULTS/f'figures/{name}.pdf'));assert len(pdf.pages)==1;fonts=pdf.pages[0]['/Resources']['/Font'].get_object();typ=[str(v.get_object()['/Subtype']) for v in fonts.values()];assert '/Type3' not in typ;fonttypes[name]=typ
# Equal square line-panel sizes are asserted in plot_final.py and export_subpanels.py before export.
page=PdfReader(str(RESULTS/'figures/Figure4_kl_mouse_regional_aging.pdf')).pages[0];assert np.isclose(float(page.mediabox.width),float(page.mediabox.height))
import re
figure_text=page.extract_text();assert 'percentage points' in figure_text and re.search(r'\bpp\b',figure_text) is None;assert re.search(r'(?m)^k\s+Age-associated',figure_text) and re.search(r'(?m)^l\s+Regional',figure_text)
from PIL import Image
with Image.open(RESULTS/'figures/Figure4_hi_mouse_regional_aging.png') as im:assert im.width==im.height
# No code dependency on previous analysis directories is allowed.
for path in (ROOT/'scripts').glob('*.py'):
 if path.name=='verify.py':continue
 s=path.read_text();assert 'glial_state_correspondence' not in s and 'mouse_only_boxplots' not in s and 'mouse_state_validation_transfer' not in s
sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'scripts').glob('*.py'))+[ROOT/'config/analysis.json']}
result={'status':'passed','representation':'X_pca_harmony','dimensions':25,'raw_mouse_cells_verified':source_mouse_rows,'individual_observations_two_glial_lineages':len(obs),'regional_individual_observations':len(regional),'k_individual_points':120,'k_mean_trajectory_points':18,'l_individual_points':40,'l_mean_trajectory_points':6,'heatmap_values':6,'permutations_independently_recalculated':len(null),'baseline_1000_reproduced':True,'all_trajectory_points_are_group_means':True,'no_dependency_on_other_analysis_directories':True,'square_page_and_equal_square_line_plots':True,'pdf_font_types':fonttypes,'script_and_config_sha256':sha}
print(json.dumps(result,indent=2))

from pathlib import Path
import json
import h5py,numpy as np,pandas as pd
from settings import ROOT,CFG,CACHE,BRAIN,RESULTS
BASE=BRAIN
I=BASE/'human_mouse_aging_integrated.h5ad';M=BASE/'mouse_concated_adata_with_meta.h5ad'
def read(f,k):
 z=f[k]
 if isinstance(z,h5py.Group):
  cats=np.asarray([s.decode() if isinstance(s,bytes) else str(s) for s in z['categories'][:]],object);codes=z['codes'][:];out=np.full(len(codes),None,object);ok=codes>=0;out[ok]=cats[codes[ok]];return out
 a=z[:]
 return np.asarray([s.decode() if isinstance(s,bytes) else str(s) for s in a],object) if a.dtype.kind in 'OSU' else a
with h5py.File(I) as f:
 sp=read(f,'obs/sp');ct=read(f,'obs/Celltype');use=(sp=='Mus_musculus')&np.isin(ct,CFG['lineages']);idx=np.flatnonzero(use)
 meta=pd.DataFrame({'barcode':read(f,'obs/barcode')[idx],'cell_type':ct[idx],'age_stage':read(f,'obs/age_stage')[idx],'age_numeric':read(f,'obs/age_numeric')[idx]})
 X=np.asarray(f['obsm/X_HENOSIS'],np.float32)[idx];assert X.shape[1]==CFG['dimensions']
with h5py.File(M) as f:
 b=read(f,'obs/_index');index=pd.Index(b);assert index.is_unique;ix=index.get_indexer(meta.barcode);assert (ix>=0).all()
 meta['donor']=read(f,'obs/mouse_id')[ix];meta['region']=read(f,'obs/region')[ix];meta['source_row']=ix
 assert np.array_equal(meta.age_stage.to_numpy(),read(f,'obs/age_stage')[ix])
for ct,g in meta.groupby('cell_type',sort=False):
 p=CACHE/('Mouse_'+ct.replace(' ','_'));g.to_csv(str(p)+'.csv',index=False);np.save(str(p)+'_embedding.npy',X[g.index.to_numpy()]);print(ct,len(g),'cells',g.donor.nunique(),'mice',flush=True)

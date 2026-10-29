#!/usr/bin/env python3
"""Recompute mouse-to-human transfer using the k=101 cosine-distance kNN."""
from pathlib import Path
import os
import h5py,numpy as np,pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
HERE=Path(__file__).resolve();PAPER=HERE.parents[4];DATA=Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/xenium/brain_integrated.h5ad';OUT=Path(os.environ.get('FIGURE4_OUTPUT_DIR',PAPER/'review_output/Figure4'))/'results/xenium/reproduced_transfer';OUT.mkdir(parents=True,exist_ok=True)
def read(g,key):
 x=g[key]
 if isinstance(x,h5py.Group):
  cats=x['categories'].asstr()[:];codes=x['codes'][:];return np.array([cats[i] if i>=0 else 'nan' for i in codes])
 return x.asstr()[:]
with h5py.File(DATA) as f:
 obs=f['obs'];sp=read(obs,'sp');ct=read(obs,'Celltype');ids=read(obs,'_index');x=normalize(np.asarray(f['obsm/X_HENOSIS'],dtype='float32'))
valid=~np.isin(ct,['Vascular/Stromal','nan','NaN']);train=(sp=='Mus_musculus')&valid;test=(sp=='homo_sapiens')&valid
shared=sorted(set(ct[train])&set(ct[test]));train &= np.isin(ct,shared);test &= np.isin(ct,shared)
print('Reference cells',train.sum(),'test cells',test.sum(),'classes',shared,flush=True)
model=KNeighborsClassifier(n_neighbors=101,metric='cosine',weights='distance',n_jobs=4);model.fit(x[train],ct[train]);pred=model.predict(x[test]);true=ct[test]
pd.DataFrame({'cell_id':ids[test],'true_celltype':true,'predicted_celltype':pred}).to_csv(OUT/'human_cell_predictions_k101.csv.gz',index=False,compression='gzip')
counts=confusion_matrix(true,pred,labels=shared);fr=counts/counts.sum(axis=1,keepdims=True);pd.DataFrame(fr,index=shared,columns=shared).to_csv(OUT/'confusion_matrix_k101_fraction.csv');pd.DataFrame(counts,index=shared,columns=shared).to_csv(OUT/'confusion_matrix_k101_counts.csv');pd.DataFrame({'celltype':shared,'n_true':counts.sum(axis=1),'n_correct':counts.diagonal(),'recall':fr.diagonal()}).to_csv(OUT/'recall_k101.csv',index=False)
fig,ax=plt.subplots(figsize=(6,5));sns.heatmap(fr,annot=True,fmt='.2f',cmap='Blues',vmin=0,vmax=1,xticklabels=shared,yticklabels=shared,ax=ax);ax.set(xlabel='Predicted cell type',ylabel='Annotated cell type');fig.tight_layout();plt.rcParams['pdf.fonttype']=42;fig.savefig(OUT/'confusion_matrix_k101.pdf',bbox_inches='tight');plt.close(fig)
print('Computed k=101 transfer; result in',OUT,flush=True)

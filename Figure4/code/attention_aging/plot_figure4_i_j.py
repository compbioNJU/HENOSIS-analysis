#!/usr/bin/env python3
"""Plot Figure 4i/j from attention tables and extracted expression source data."""
from pathlib import Path
import os
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
PAPER=Path(__file__).resolve().parents[4]; OUT=Path(os.environ.get('FIGURE4_OUTPUT_DIR',PAPER/'review_output/Figure4'))/'results/attention_aging';SOURCE=OUT/'source_data';OUT.mkdir(parents=True,exist_ok=True)
from plotting_helpers import violinplot_reference, STAGE_PALETTE, set_plot_style
set_plot_style()
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
for species,gene in [('mouse','Olig2'),('human','OLIG2')]:
 cell=pd.read_csv(SOURCE/'Fig4j'/f'{species}_{gene}_OPC_violin_sampled_cells.csv').rename(columns={'age_stage':'plot_group','expression_used_in_plot':'expr_log'})
 donor=pd.read_csv(SOURCE/'Fig4j'/f'{species}_{gene}_OPC_sample_summary.csv').rename(columns={'age_stage':'plot_group'})
 fig,ax=plt.subplots(figsize=(2.5,2.4))
 violinplot_reference(ax=ax,cell_df=cell,sample_df=donor,x_col='plot_group',y_col_cell='expr_log',y_col_sample='mean_expr_log',order=['young','middle','old'],palette=STAGE_PALETTE,title=gene,ylabel='Log expression',xlabel='age_stage')
 fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(OUT/f'{species}_{gene}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
# Recreate the two ranked heatmaps using the rank and score table supplied by user.
fig,axes=plt.subplots(1,2,figsize=(9,6));cmap=plt.cm.Blues.copy();cmap.set_bad('#EDEDED')
for ax,species in zip(axes,['mouse','human']):
 raw_name={'mouse':'Mus_musculus','human':'homo_sapiens'}[species]+'_attention_score_topk_freq.csv'
 df=pd.read_csv(Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/attention_aging'/raw_name)
 df=df.sort_values(['cell_type','age_stage','score'],ascending=[True,True,False])
 df['rank']=df.groupby(['cell_type','age_stage'])['score'].rank(method='first',ascending=False)
 ref=pd.read_csv(Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/attention_aging'/f'{species}_age_stage_with_rank.csv')
 compared=df.merge(ref,on=['gene','cell_type','age_stage'],suffixes=('_derived','_reference'),validate='one_to_one')
 assert len(compared)==len(df)==len(ref) and np.allclose(compared.score_derived,compared.score_reference) and np.array_equal(compared['rank_derived'],compared['rank_reference'])
 table_dir=SOURCE/'Fig4i';table_dir.mkdir(parents=True,exist_ok=True);df.to_csv(table_dir/f'{species}_age_stage_rank_recomputed.csv',index=False)
 df=df[(df.cell_type=='OPC')&(df['rank']<=10)]
 order=[]
 for age in ['young','middle','old']:
  for gene in df[df.age_stage.eq(age)].sort_values(['rank','score'],ascending=[True,False]).gene:
   if gene not in order:order.append(gene)
 pivot=df.pivot(index='gene',columns='age_stage',values='score').reindex(index=order,columns=['young','middle','old']); ranks=df.pivot(index='gene',columns='age_stage',values='rank').reindex(index=order,columns=['young','middle','old'])
 im=ax.imshow(pivot.to_numpy(float),cmap=cmap,vmin=0,vmax=1,aspect='auto');ax.set_title(species.capitalize());ax.set_yticks(range(len(order)),order);ax.set_xticks(range(3),['young','middle','old']);ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(order),1),minor=True);ax.grid(which='minor',color='white',lw=1.3);ax.tick_params(which='minor',bottom=False,left=False)
 for (i,j),v in np.ndenumerate(ranks.to_numpy(float)):
  if np.isfinite(v):ax.text(j,i,str(int(v)),ha='center',va='center',color='white' if pivot.iloc[i,j]>.5 else '#263d55')
 for label in ax.get_yticklabels():
  label.set_fontstyle('italic')
  if label.get_text().upper() in ['OLIG2','SOX2']:label.set_color('#e83265')
fig.colorbar(im,ax=axes,fraction=.025,pad=.05,label='Attention score');fig.savefig(OUT/'OPC_ranked_tile_heatmap.pdf',bbox_inches='tight');fig.savefig(OUT/'OPC_ranked_tile_heatmap.png',dpi=300,bbox_inches='tight');plt.close(fig)
print('Plotted two expression panels and OPC ranked heatmap:',OUT)

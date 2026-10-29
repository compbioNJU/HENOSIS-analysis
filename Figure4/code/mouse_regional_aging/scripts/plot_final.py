#!/usr/bin/env python3
"""Final k–l design: individual dots and age-group mean trajectories, from local results only."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from settings import ROOT,CFG,RESULTS
from vector_heatmap import draw_vector_heatmap
T=RESULTS/'tables';F=RESULTS/'figures';F.mkdir(exist_ok=True)
regional=pd.read_csv(T/'individual_region_occupancy.csv',dtype={'donor':str});tests=pd.read_csv(T/'mouse_full_pipeline_twosided.csv')
cts=CFG['glial_lineages'];regions=CFG['regions'];stages=['young','middle','old'];regioncolors=CFG['plot_h_colors'];ctcolors=CFG['plot_i_colors'];rng=np.random.default_rng(CFG['seed'])
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
# Square page with three identical square line-plot rectangles.
fig=plt.figure(figsize=(10.4,10.4));side=.33
h_rectangles=[[.105,.56,side,side],[.60,.56,side,side]]
i_rectangle=[.60,.13,side,side];heat_rectangle=[.105,.13,side,side];curve_axes=[]
fig.text(.045,.97,'k',fontsize=17,weight='bold');fig.text(.105,.97,'Age-associated changes across brain regions',fontsize=12,weight='bold');fig.legend(handles=[Line2D([],[],marker='o',color=regioncolors[r],label=r,lw=1.8,markersize=5) for r in regions],loc='upper center',bbox_to_anchor=(.56,.95),ncol=3,frameon=False,fontsize=9)
region_means=[]
for col,ct in enumerate(cts):
 ax=fig.add_axes(h_rectangles[col]);curve_axes.append(ax);g=regional[regional.cell_type.eq(ct)]
 for k,r in enumerate(regions):
  mean_x=[];mean_y=[]
  for j,st in enumerate(stages):
   d=g[g.age_stage.eq(st)&g.region.eq(r)].sort_values('donor');v=d.occupancy_pct.to_numpy();assert len(v)==[4,7,9][j];pos=j+(k-1)*.22;ax.scatter(pos+rng.uniform(-.045,.045,len(v)),v,c=regioncolors[r],s=64,alpha=.65,edgecolors='none',linewidths=0,zorder=2);mean_x.append(pos);mean_y.append(v.mean());region_means.append(dict(cell_type=ct,region=r,age_stage=st,n_mice=len(v),mean_high_age_score_cells_pct=v.mean()))
  ax.plot(mean_x,mean_y,c=regioncolors[r],lw=2.1,zorder=3);ax.scatter(mean_x,mean_y,c=regioncolors[r],s=121,edgecolors='none',linewidths=0,zorder=4)
 ax.set_title(ct,fontsize=11,pad=10);ax.set_ylim(0,83);ax.set_yticks([0,20,40,60,80]);ax.set_xticks([0,1,2],['Young\n(n=4)','Middle\n(n=7)','Older\n(n=9)']);ax.grid(axis='y',c='#edf0f1',lw=.7);ax.set_axisbelow(True)
 if col==0:ax.set_ylabel('Cells with high age scores (%)')
 else:ax.tick_params(labelleft=False)
# Bottom l: descriptive heatmap and the combined within-mouse regional contrast.
fig.text(.045,.505,'l',fontsize=17,weight='bold');fig.text(.105,.505,'Regional age-associated changes',fontsize=12,weight='bold');holder=fig.add_axes(heat_rectangle);holder.axis('off');heatrows=[]
for ct in cts:
 for r in regions:
  g=regional[regional.cell_type.eq(ct)&regional.region.eq(r)];young=g[g.stage.eq(0)].occupancy_pct.mean();older=g[g.stage.eq(2)].occupancy_pct.mean();heatrows.append(dict(cell_type=ct,region=r,young_mean_pct=young,older_mean_pct=older,older_minus_young_pp=older-young))
heat=pd.DataFrame(heatrows);matrix=heat.pivot(index='region',columns='cell_type',values='older_minus_young_pp').loc[regions,cts].to_numpy();ha=holder.inset_axes([.03,.20,.92,.7]);im=draw_vector_heatmap(ha,matrix);ha.set_yticks(range(3),regions,fontsize=8);ha.set_xticks(range(2),['ODC','Microglia'],fontsize=8);ha.tick_params(length=0);ha.set_title('Older − young\nchange (percentage points)',fontsize=9,pad=9)
for (r,c),v in np.ndenumerate(matrix):ha.text(c,r,f'+{v:.1f}',ha='center',va='center',color='white' if v>20 else '#293d47',fontsize=9)
ha.set_xticks([-.5,.5,1.5],minor=True);ha.set_yticks([-.5,.5,1.5,2.5],minor=True);ha.grid(which='minor',c='white',lw=1.2);ha.tick_params(which='minor',bottom=False,left=False);cbax=holder.inset_axes([.03,.025,.92,.045]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal');cb.solids.set_rasterized(False);cb.set_ticks([0,20,40]);cb.ax.tick_params(labelsize=7,length=2)
ax=fig.add_axes(i_rectangle);curve_axes.append(ax);contrasts=[];contrastmeans=[];statrows=[]
for k,ct in enumerate(cts):
 g=regional[regional.cell_type.eq(ct)];wide=g.pivot(index=['donor','age_stage','stage'],columns='region',values='occupancy_pct').reset_index();wide['CC_ACO_minus_CTX_pp']=wide['CC/ACO']-wide['CTX'];wide['cell_type']=ct;contrasts.append(wide);mx=[];my=[]
 for j,st in enumerate(stages):
  v=wide[wide.age_stage.eq(st)].CC_ACO_minus_CTX_pp.to_numpy();pos=j+(k-.5)*.18;ax.scatter(pos+rng.uniform(-.055,.055,len(v)),v,c=ctcolors[ct],s=64,alpha=.65,edgecolors='none',linewidths=0,zorder=2);mx.append(pos);my.append(v.mean());contrastmeans.append(dict(cell_type=ct,age_stage=st,n_mice=len(v),mean_CC_ACO_minus_CTX_pp=v.mean()))
 ax.plot(mx,my,c=ctcolors[ct],lw=2.3,zorder=3);ax.scatter(mx,my,c=ctcolors[ct],s=121,edgecolors='none',linewidths=0,zorder=4)
 delta=my[2]-my[0];r=tests[tests.cell_type.eq(ct)&tests.metric.eq('region_interaction')].iloc[0];assert np.isclose(delta,100*r.full_cell_effect);statrows.append(dict(cell_type=ct,full_cell_older_minus_young_regional_contrast_pp=delta,subsample_observed_contrast_pp=100*r.subsample_observed,p_two_sided=r.p_two,q_two_exploratory=r.q_BH_interactions_two));label=ct;ax.text(.025,.91-k*.075,f'{label}: +{delta:.1f} (q = {r.q_BH_interactions_two:.4f})',transform=ax.transAxes,va='top',fontsize=8.5,color=ctcolors[ct])
ax.text(.025,.98,'Older − young change in regional difference',transform=ax.transAxes,va='top',fontsize=8,color='#354b59')
ax.set_ylim(-17,55);ax.set_yticks([-10,0,10,20,30,40,50]);ax.axhline(0,color='#99aab4',lw=.8,ls='--');ax.set_xticks([0,1,2],['Young\n(n=4)','Middle\n(n=7)','Older\n(n=9)']);ax.set_ylabel('CC/ACO − CTX (percentage points)');ax.grid(axis='y',c='#edf0f1',lw=.7);ax.set_axisbelow(True);ax.legend(handles=[Line2D([],[],marker='o',color=ctcolors[c],lw=2,markersize=5,label=c) for c in cts],loc='upper center',bbox_to_anchor=(.54,1.16),ncol=2,frameon=False,fontsize=8.5)
fig.text(.095,.025,'Small translucent dots: individual mice. Larger points and connecting lines: age-group means, not individual longitudinal trajectories.\nHeatmap: descriptive full-cell older−young changes. Statistical definitions and fixed-subsample permutation tests are described in the legend.',fontsize=8,color='#526570')
fig.canvas.draw()
plot_sizes=[(a.get_position().width*fig.get_figwidth(),a.get_position().height*fig.get_figheight()) for a in curve_axes]
assert len(plot_sizes)==3 and np.allclose(plot_sizes,np.tile(plot_sizes[0],(3,1))) and np.allclose(np.array(plot_sizes)[:,0],np.array(plot_sizes)[:,1])
# Do not use tight bounding-box cropping: preserve the exact square page.
for ext in ['pdf','png','svg']:
 for stem in ['Figure4_kl_mouse_regional_aging','Figure4_hi_mouse_regional_aging']:
  fig.savefig(F/f'{stem}.{ext}',dpi=500)
plt.close(fig)
# Frozen source tables include the exact mean points used in every connecting line.
regional.rename(columns={'occupancy_pct':'cells_with_high_age_scores_pct'}).to_csv(T/'figure_k_individuals.csv',index=False);pd.DataFrame(region_means).to_csv(T/'figure_k_region_group_means.csv',index=False);pd.concat(contrasts).to_csv(T/'figure_l_individual_contrasts.csv',index=False);pd.DataFrame(contrastmeans).to_csv(T/'figure_l_lineage_group_means.csv',index=False);heat.to_csv(T/'figure_l_regional_heatmap.csv',index=False);pd.DataFrame(statrows).to_csv(T/'figure_l_regional_statistics.csv',index=False)
with pd.ExcelWriter(T/'Figure4_kl_source_data.xlsx') as writer:
 pd.read_csv(T/'figure_k_individuals.csv',dtype={'donor':str}).to_excel(writer,sheet_name='k_individuals',index=False);pd.DataFrame(region_means).to_excel(writer,sheet_name='k_group_mean_trajectories',index=False);pd.concat(contrasts).to_excel(writer,sheet_name='l_individual_contrasts',index=False);pd.DataFrame(contrastmeans).to_excel(writer,sheet_name='l_group_mean_trajectories',index=False);heat.to_excel(writer,sheet_name='l_descriptive_heatmap',index=False);pd.DataFrame(statrows).to_excel(writer,sheet_name='l_statistics',index=False)
print('Final Figure 4 k–l and source workbook generated',flush=True)

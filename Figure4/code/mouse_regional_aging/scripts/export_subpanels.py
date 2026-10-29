#!/usr/bin/env python3
"""Export four independently drawn Figure 4 k–l subpanels, preserving editable text."""
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypdf import PdfReader
from settings import ROOT,CFG,RESULTS
from vector_heatmap import draw_vector_heatmap
T=RESULTS/'tables';OUT=RESULTS/'figures/subpanels';OUT.mkdir(parents=True,exist_ok=True)
regional=pd.read_csv(T/'individual_region_occupancy.csv',dtype={'donor':str});tests=pd.read_csv(T/'mouse_full_pipeline_twosided.csv');source_names=['individual_region_occupancy.csv','mouse_full_pipeline_twosided.csv'];before={n:hashlib.sha256((T/n).read_bytes()).hexdigest() for n in source_names}
cts=CFG['glial_lineages'];regions=CFG['regions'];stages=['young','middle','old'];rc=CFG['plot_h_colors'];cc=CFG['plot_i_colors'];rng=np.random.default_rng(CFG['seed']);manifest=[]
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name,ax,n_points,n_means):
 fig.canvas.draw();size=[float(ax.get_position().width*fig.get_figwidth()),float(ax.get_position().height*fig.get_figheight())]
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=500)
 reader=PdfReader(str(OUT/f'{name}.pdf'));assert len(reader.pages)==1;page=reader.pages[0];assert np.isclose(float(page.mediabox.width),float(page.mediabox.height));fonts=page['/Resources']['/Font'].get_object();types=[str(f.get_object()['/Subtype']) for f in fonts.values()];assert '/Type3' not in types
 manifest.append({'name':name,'pdf':str((OUT/f'{name}.pdf').relative_to(RESULTS)),'pdf_pages':1,'page_inches':[5.2,5.2],'plot_inches':size,'individual_points':n_points,'group_mean_points':n_means,'font_types':types});plt.close(fig)
for ct in cts:
 fig=plt.figure(figsize=(5.2,5.2));ax=fig.add_axes([.20,.17,.66,.66]);g=regional[regional.cell_type.eq(ct)]
 for k,r in enumerate(regions):
  mx=[];my=[]
  for j,st in enumerate(stages):
   d=g[g.region.eq(r)&g.age_stage.eq(st)].sort_values('donor');v=d.occupancy_pct.to_numpy();pos=j+(k-1)*.22;ax.scatter(pos+rng.uniform(-.045,.045,len(v)),v,c=rc[r],s=64,alpha=.65,edgecolors='none',linewidths=0,zorder=2);mx.append(pos);my.append(v.mean())
  ax.plot(mx,my,c=rc[r],lw=2.1,zorder=3);ax.scatter(mx,my,c=rc[r],s=121,edgecolors='none',linewidths=0,zorder=4)
 ax.set_title(ct,fontsize=11,pad=10);ax.set_ylim(0,83);ax.set_yticks([0,20,40,60,80]);ax.set_xticks([0,1,2],['Young\n(n=4)','Middle\n(n=7)','Older\n(n=9)']);ax.set_ylabel('Cells with high age scores (%)');ax.grid(axis='y',c='#edf0f1',lw=.7);ax.set_axisbelow(True);ax.legend(handles=[Line2D([],[],marker='o',color=rc[r],lw=1.8,markersize=5,label=r) for r in regions],loc='upper center',bbox_to_anchor=(.50,1.22),ncol=3,frameon=False,fontsize=8,columnspacing=.9,handlelength=1.8)
 save(fig,'k_'+ct,ax,60,9)
# Standalone heatmap includes its complete color scale and unit label.
fig=plt.figure(figsize=(5.2,5.2));ax=fig.add_axes([.22,.27,.65,.56]);rows=[]
for ct in cts:
 for r in regions:
  g=regional[regional.cell_type.eq(ct)&regional.region.eq(r)];rows.append({'cell_type':ct,'region':r,'change':g[g.stage.eq(2)].occupancy_pct.mean()-g[g.stage.eq(0)].occupancy_pct.mean()})
heat=pd.DataFrame(rows).pivot(index='region',columns='cell_type',values='change').loc[regions,cts].to_numpy();im=draw_vector_heatmap(ax,heat);ax.set_yticks(range(3),regions,fontsize=9);ax.set_xticks(range(2),['Oligodendrocyte','Microglia'],fontsize=8);ax.tick_params(length=0);ax.set_title('Older − young change\n(percentage points)',fontsize=11,pad=14)
for (r,c),v in np.ndenumerate(heat):ax.text(c,r,f'+{v:.1f}',ha='center',va='center',fontsize=12,color='white' if v>20 else '#293d47')
ax.set_xticks([-.5,.5,1.5],minor=True);ax.set_yticks([-.5,.5,1.5,2.5],minor=True);ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='minor',bottom=False,left=False);cbax=fig.add_axes([.22,.14,.65,.035]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal');cb.solids.set_rasterized(False);cb.set_ticks([0,20,40]);cb.ax.tick_params(labelsize=8,length=2)
save(fig,'l_Regional_change_heatmap',ax,0,0)
# Standalone merged contrast retains both lineages and original statistic labels.
fig=plt.figure(figsize=(5.2,5.2));ax=fig.add_axes([.20,.17,.66,.66])
for k,ct in enumerate(cts):
 g=regional[regional.cell_type.eq(ct)];wide=g.pivot(index=['donor','age_stage','stage'],columns='region',values='occupancy_pct').reset_index();wide['contrast']=wide['CC/ACO']-wide['CTX'];mx=[];my=[]
 for j,st in enumerate(stages):
  v=wide[wide.age_stage.eq(st)].contrast.to_numpy();pos=j+(k-.5)*.18;ax.scatter(pos+rng.uniform(-.055,.055,len(v)),v,c=cc[ct],s=64,alpha=.65,edgecolors='none',linewidths=0,zorder=2);mx.append(pos);my.append(v.mean())
 ax.plot(mx,my,c=cc[ct],lw=2.3,zorder=3);ax.scatter(mx,my,c=cc[ct],s=121,edgecolors='none',linewidths=0,zorder=4);r=tests[tests.cell_type.eq(ct)&tests.metric.eq('region_interaction')].iloc[0];assert np.isclose(my[2]-my[0],100*r.full_cell_effect);ax.text(.025,.91-k*.075,f'{ct}: +{my[2]-my[0]:.1f} (q = {r.q_BH_interactions_two:.4f})',transform=ax.transAxes,va='top',fontsize=8.5,color=cc[ct])
ax.text(.025,.98,'Older − young change in regional difference',transform=ax.transAxes,va='top',fontsize=8,color='#354b59');ax.set_ylim(-17,55);ax.set_yticks([-10,0,10,20,30,40,50]);ax.axhline(0,c='#99aab4',lw=.8,ls='--');ax.set_xticks([0,1,2],['Young\n(n=4)','Middle\n(n=7)','Older\n(n=9)']);ax.set_ylabel('CC/ACO − CTX (percentage points)');ax.grid(axis='y',c='#edf0f1',lw=.7);ax.set_axisbelow(True);ax.legend(handles=[Line2D([],[],marker='o',color=cc[c],lw=2,markersize=5,label=c) for c in cts],loc='upper center',bbox_to_anchor=(.50,1.16),ncol=2,frameon=False,fontsize=8.5,columnspacing=1.)
save(fig,'l_CC_ACO_minus_CTX',ax,40,6)
curves=[r['plot_inches'] for r in manifest if r['name']!='l_Regional_change_heatmap'];assert np.allclose(curves,np.tile([3.432,3.432],(3,1)))
assert before=={n:hashlib.sha256((T/n).read_bytes()).hexdigest() for n in source_names}
(OUT/'export_manifest.json').write_text(json.dumps({'status':'passed','panels':manifest,'all_line_plot_sizes_identical':True,'statistics_unchanged':True},indent=2)+'\n')
print('Saved four separate square-page PDFs, PNGs and SVGs to',OUT)

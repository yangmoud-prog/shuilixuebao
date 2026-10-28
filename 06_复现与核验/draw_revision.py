from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
HERE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('RWEC_SOURCE_ROOT',str(HERE/'07_接续资料/source_snapshot')))
DEST=HERE
OUT=DEST/'05_修订图件';OUT.mkdir(exist_ok=True)
SRC=ROOT/'Code/figure code/paper_final_20260831/figure_data'
mpl.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.serif':['Times New Roman','SimSun'],'font.size':9,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in','figure.facecolor':'white'})
COL=['#0072B2','#009E73','#D55E00']
def axes_style(ax):
    ax.tick_params(top=True,right=True,width=.6,length=3)
    ax.grid(alpha=.18,lw=.4)
def distance(ax):
    axes_style(ax);ax.set_xlim(7,123);ax.set_xticks([10,30,50,70,90,120]);ax.set_xlabel('距离 (km)')
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=300)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
d=pd.read_csv(SRC/'Fig_04_lmcs_k_n_tau_patterns_source_data.csv')
d=d[d.model.eq('LMCS') & d.geometry.isin(['G1','G2','G3'])]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.65))
for ax,(p,label) in zip(axs,[('k','k'),('n','n'),('tau_h','τ (h)')]):
    for j,g in enumerate(['G1','G2','G3']):
        q=d[d.geometry.eq(g)].sort_values('distance_km');assert q.distance_km.tolist()==list(range(10,121,10))
        ax.plot(q.distance_km,q[p],color=COL[j],marker=['o','s','^'][j],ms=3,lw=1,label='C'+str(j+1))
    distance(ax);ax.set_ylabel(label);ax.text(0,1.03,'('+chr(97+list(axs).index(ax))+')',transform=ax.transAxes)
axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.49),ncol=3,frameon=False)
fig.subplots_adjust(left=.07,right=.985,top=.9,bottom=.3,wspace=.37)
save(fig,'图4_120km刻度修订')
sd=pd.read_csv(SRC/'Fig_06_slope_parameter_2d_source_data.csv')
rd=pd.read_csv(SRC/'Fig_07_roughness_parameter_2d_source_data.csv')
fig,axs=plt.subplots(2,3,figsize=(7.2,4.7))
for r,data,groups,labels in [(0,sd,['S0001_N035','S0002_N035','S0005_N035'],['比降 0.10‰','比降 0.20‰','比降 0.50‰']),(1,rd,['S0002_N025','S0002_N035','S0002_N050'],['糙率 0.025','糙率 0.035','糙率 0.050'])]:
    for c,p in enumerate(['k','n','tau_h']):
        ax=axs[r,c]
        for j,g in enumerate(groups):
            q=data[data.geometry.eq(g)&data.model.eq('LMCS')].sort_values('distance_km');assert q.distance_km.tolist()==list(range(10,121,10))
            ax.plot(q.distance_km,q[p],color=COL[j],marker=['o','s','^'][j],ms=2.8,lw=.9,label=labels[j])
        distance(ax);ax.set_ylabel(['k','n','τ (h)'][c])
    axs[r,0].legend(fontsize=7,loc='upper left',frameon=False)
    axs[r,0].text(-.23,1.08,['(a) 固定糙率 0.035','(b) 固定比降 0.20‰'][r],transform=axs[r,0].transAxes)
fig.subplots_adjust(left=.075,right=.985,top=.92,bottom=.12,wspace=.39,hspace=.60)
save(fig,'图6_120km刻度修订')
rat=pd.read_csv(DEST/'04_补充分析/36组_tau_Tp_残差与比值.csv')
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6))
for j,g in enumerate(['C1','C2','C3']):
    q=rat[rat.channel.eq(g)].sort_values('distance_km')
    axs[0].scatter(q.Tp_h,q.tau_h,c=COL[j],s=16,marker=['o','s','^'][j],label=g)
    axs[1].plot(q.distance_km,q.tau_minus_Tp_h,c=COL[j],ms=3,marker=['o','s','^'][j],lw=.8)
    axs[2].plot(q.distance_km,q.tau_div_Tp,c=COL[j],ms=3,marker=['o','s','^'][j],lw=.8)
limit=max(rat.Tp_h.max(),rat.tau_h.max())+1
axs[0].plot([0,limit],[0,limit],c='.5',ls='--',lw=.6);axs[0].set(xlim=(0,limit),ylim=(0,limit),xlabel='Tp (h)',ylabel='τ (h)')
axs[0].text(.04,.93,'r = 0.986',transform=axs[0].transAxes,fontsize=8)
axs[0].legend(loc='lower right',fontsize=7,frameon=False);axes_style(axs[0])
distance(axs[1]);axs[1].axhline(0,c='.5',ls='--',lw=.6);axs[1].set_ylabel('τ − Tp (h)')
distance(axs[2]);axs[2].axhline(1,c='.5',ls='--',lw=.6);axs[2].set_ylabel('τ / Tp');axs[2].set_ylim(.35,1.05)
for j,ax in enumerate(axs):ax.text(0,1.035,'('+chr(97+j)+')',transform=ax.transAxes)
fig.subplots_adjust(left=.07,right=.985,top=.89,bottom=.24,wspace=.46)
save(fig,'图7_残差与比值补充')
(OUT/'图件核验.json').write_text(json.dumps({'fonts':['Times New Roman','SimSun'],'source_figure4_rows':len(d),'figure4_geometry_distance_pairs':36,'figure6_rows':[len(sd),len(rd)],'maximum_tick_km':120,'source_changes':'仅显示范围刻度和排版 数据未改动','figure7_combinations':len(rat),'exports':['png 300dpi','editable svg','pdf TrueType']},ensure_ascii=False,indent=2),encoding='utf-8')
print('3 figures exported')

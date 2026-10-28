from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import os
HERE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('RWEC_SOURCE_ROOT',str(HERE/'07_接续资料/source_snapshot')))
DEST=HERE
OUT = DEST/'04_补充分析'
OUT.mkdir(exist_ok=True)
SOURCES=[]
def read(p,**kw):
    p=Path(p); SOURCES.append({'文件':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return pd.read_csv(p,**kw)

fig=ROOT/'Code/figure code/paper_final_20260831/figure_data'
pars=read(fig/'Fig_04_lmcs_k_n_tau_patterns_source_data.csv')
pars=pars.loc[pars.model.eq('LMCS') & pars.geometry.isin(['G1','G2','G3'])].copy()
raw=ROOT/'HECRAS/random_inflow_generalization_100/05_extracted_flow/corrected_full100_1km_warm72h'
rows=[]
for geom in ['G1','G2','G3']:
    d=read(raw/f'{geom}_hydrographs.csv')
    d=d[d.split.eq('test')].sort_values('time_h')
    for (dist,event), g in d.groupby(['distance_km','event_id']):
        t=g.time_h.to_numpy();qi=g.Q_in_m3s.to_numpy();qo=g.Q_out_m3s.to_numpy()
        rows.append({'geometry':geom,'distance_km':dist,'event_id':event,'Tp_h':float(t[np.argmax(qo)]-t[np.argmax(qi)]),'in_peak_t_h':float(t[np.argmax(qi)]),'out_peak_t_h':float(t[np.argmax(qo)])})
events=pd.DataFrame(rows)
assert len(events)==1080
events.to_csv(OUT/'HEC_1080个测试过程_参照洪峰滞时.csv',index=False,encoding='utf-8-sig')
lag=events.groupby(['geometry','distance_km']).Tp_h.median().reset_index()
rat=pars[['geometry','distance_km','tau_h','k','n']].merge(lag,on=['geometry','distance_km'],validate='1:1')
assert len(rat)==36 and (rat.Tp_h>0).all()
rat['tau_minus_Tp_h']=rat.tau_h-rat.Tp_h
rat['tau_div_Tp']=rat.tau_h/rat.Tp_h
rat['channel']=rat.geometry.map({'G1':'C1','G2':'C2','G3':'C3'})
rat.to_csv(OUT/'36组_tau_Tp_残差与比值.csv',index=False,encoding='utf-8-sig')
stats=[]
for label,g in [('全部36组',rat)]+list(rat.groupby('channel')):
    for col in ['tau_minus_Tp_h','tau_div_Tp']:
        x=g[col]
        stats.append({'分组':label,'指标':col,'n':len(x),'mean':x.mean(),'median':x.median(),'sd':x.std(ddof=1),'min':x.min(),'max':x.max(),'q25':x.quantile(.25),'q75':x.quantile(.75)})
group=pd.DataFrame(stats);group.to_csv(OUT/'tau_Tp_按河槽分组统计.csv',index=False,encoding='utf-8-sig')
rat.groupby('distance_km')[['tau_minus_Tp_h','tau_div_Tp']].agg(['mean','median','std','min','max']).to_csv(OUT/'tau_Tp_按距离分组统计.csv',encoding='utf-8-sig')

sun=ROOT/'Result/多场率定/数据/笋溪河_F08上游洪量配平数据直跑_20260920/model_results'
sd=read(sun/'逐时三模型模拟.csv')
natural=[]
for (event,model),g in sd[sd['数据用途'].eq('测试')].groupby(['事件编号','模型']):
    g=g.sort_values('step');t=g.step.to_numpy(float);o=g['实测下游流量_m3s'].to_numpy(float);s=g['模拟下游流量_m3s'].to_numpy(float);i=g['上游流量_m3s'].to_numpy(float)
    peak=int(np.argmax(o)); peaks,_=find_peaks(o,prominence=.1*np.ptp(o),distance=3)
    others=[int(v) for v in peaks if v!=peak]
    left=max([v for v in others if v<peak],default=None)
    right=min([v for v in others if v>peak],default=None)
    lo=left+int(np.argmin(o[left:peak+1])) if left is not None else 0
    hi=peak+int(np.argmin(o[peak:right+1])) if right is not None else len(o)-1
    matched=lo+int(np.argmax(s[lo:hi+1]))
    natural.append({'river':'笋溪河','event_id':event,'model':model,'n':len(g),'NSE':1-np.sum((s-o)**2)/np.sum((o-o.mean())**2),'RMSE_m3s':np.sqrt(np.mean((s-o)**2)),'Qpeak_obs_m3s':o.max(),'Qpeak_sim_m3s':s.max(),'peak_error_percent':100*(s.max()-o.max())/o.max(),'peak_time_error_h':t[np.argmax(s)]-t[np.argmax(o)],'matched_window_start_h':t[lo],'matched_window_end_h':t[hi],'matched_peak_sim_m3s':s[matched],'matched_peak_error_percent':100*(s[matched]-o[peak])/o[peak],'matched_peak_time_error_h':t[matched]-t[peak],'Tp_ref_h':t[np.argmax(o)]-t[np.argmax(i)],'Tp_sim_h':t[np.argmax(s)]-t[np.argmax(i)],'ties_obs':int(np.sum(o==o.max())),'ties_sim':int(np.sum(s==s.max()))})
natural=pd.DataFrame(natural)
natural['abs_peak_error_percent']=natural.peak_error_percent.abs();natural['abs_peak_time_error_h']=natural.peak_time_error_h.abs()
natural.to_csv(OUT/'笋溪河_逐场洪峰误差_已核实版本.csv',index=False,encoding='utf-8-sig')
natural.groupby('model')[['NSE','RMSE_m3s','abs_peak_error_percent','abs_peak_time_error_h']].mean().to_csv(OUT/'笋溪河_测试期洪峰指标均值.csv',encoding='utf-8-sig')
metrics=read(ROOT/'HECRAS/random_inflow_generalization_100/08_summary/042_gpt_key_materials_corrected_full100/data/test_event_metrics.csv')
mt=metrics[metrics.model.eq('LMCS') & metrics.geometry.isin(['G1','G2','G3'])].copy()
dec=events.merge(mt[['geometry','distance_km','event_id','peak_time_error_h']],on=['geometry','distance_km','event_id'],validate='1:1').merge(pars[['geometry','distance_km','tau_h']],on=['geometry','distance_km'],validate='m:1')
dec['Tp_sim_h']=dec.Tp_h+dec.peak_time_error_h
dec['storage_adjustment_h']=dec.Tp_sim_h-dec.tau_h
dec['translation_offset_h']=dec.tau_h-dec.Tp_h
assert np.max(abs(dec.translation_offset_h+dec.storage_adjustment_h-dec.peak_time_error_h))<1e-9
dec.to_csv(OUT/'HEC_逐场传播调蓄峰时分解.csv',index=False,encoding='utf-8-sig')
dec.groupby('geometry')[['Tp_h','tau_h','Tp_sim_h','storage_adjustment_h','translation_offset_h','peak_time_error_h']].agg(['mean','median','min','max']).to_csv(OUT/'HEC_峰时分解分组统计.csv',encoding='utf-8-sig')
# The saved Qingjiang series do not reproduce the submitted RWEC RMSE.
# Keep them as an audit candidate; never merge these results into the paper.
qpath=ROOT/'Code/figure code/paper_final_20260831/final_figures/笋溪河清江_两图两子图_20260919/清江两场测试洪水过程_source_data.csv'
qd=read(qpath)
qrows=[]
for (event,model),g in qd.groupby(['event_id','model']):
    o=g.q_obs.to_numpy(float);s=g.q_sim.to_numpy(float)
    qrows.append({'event':event,'model':model,'RMSE_m3s':np.sqrt(np.mean((s-o)**2)),'status':'候选资料 不写入论文'})
pd.DataFrame(qrows).to_csv(OUT/'清江_候选资料核查_尚未匹配送审稿.csv',index=False,encoding='utf-8-sig')

summary={'n_combinations':len(rat),'n_hec_test_processes':len(events),'pearson_tau_Tp':rat.tau_h.corr(rat.Tp_h),'ratio_median':rat.tau_div_Tp.median(),'residual_all_negative':bool((rat.tau_minus_Tp_h<0).all()),'max_n_in_fig4':float(rat.n.max()),'ratio_groups':group.to_dict('records'),'sunxi':natural.to_dict('records'),'qingjiang_candidate_RWEC_RMSE':float(pd.DataFrame(qrows).query("model=='RWEC'").RMSE_m3s.mean())}
(OUT/'核查摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(SOURCES).to_csv(OUT/'来源与哈希清单.csv',index=False,encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2))

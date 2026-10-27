"""Render only measured CSV evidence. All plots carry units and source captions."""
import argparse
import csv
from pathlib import Path
import numpy as np


def load(root,name):
    with (root/name).open(newline='') as f:return list(csv.DictReader(f))


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','figure.facecolor':'white'})
    caption='Source: S-E3DATA validation only, 36 worlds; WM2, K=8; warmup end to +300 s.'
    summary=load(root,'ablation_summary.csv')
    names=['baseline','residual_off','population','courtesy_off','yield_off','ramp_adapt_off']
    labels=['WM2','Physics only','Population','Courtesy off','Yield off','Ramp adapt off']
    if summary:
        fig,ax=plt.subplots(figsize=(11,4.5))
        for arm,offset,color,label in [('mappo',-.12,'#275d89','MAPPO − native'),('ours_c1',.12,'#a74b25','Ours − native')]:
            rows=[next((r for r in summary if r['variant']==v and r['arm']==arm and r['horizon_s']=='300'),None) for v in names]
            for i,r in enumerate(rows):
                if r is None:continue
                y=float(r['slope']);lo=float(r['slope_lo95']);hi=float(r['slope_hi95'])
                ax.errorbar(i+offset,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color,capsize=3,
                            label=label if i==0 else None)
        ax.axhline(1,color='black',lw=1,ls='--',label='Ideal slope = 1')
        ax.set_xticks(range(6),labels);ax.set_ylabel('Predicted-on-real paired dJ OLS slope (unitless)')
        ax.set_xlabel('Longitudinal switch-off ablation');ax.set_title('Which WM component suppresses 300 s controller effects?')
        ax.legend(frameon=False,ncol=3);fig.text(.02,.01,caption+' Error bars: 95% world bootstrap.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));fig.savefig(root/'ablation_slopes.svg');plt.close(fig)
    timeline=load(root,'timeline_summary.csv')
    if timeline:
        fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True)
        for j,region in enumerate(('upstream','merge','ramp','downstream')):
            for i,cls in enumerate(('HDV','CAV')):
                ax=axes[i,j]
                for arm,color,label in [('mappo','#275d89','MAPPO − native'),('ours_c1','#a74b25','Ours − native')]:
                    rs=[r for r in timeline if r['region']==region and r['class']==cls and r['arm']==arm]
                    if rs:ax.plot([int(r['bin_end_s']) for r in rs],[float(r['cumulative_error']) for r in rs],label=label,color=color)
                ax.axhline(0,color='grey',lw=.6);ax.set_title(region.capitalize()+' · '+cls)
                ax.set_xlabel('Time since warmup end (s)');ax.set_ylabel('Cumulative paired dJ error (veh-s)')
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle('Where and when does paired-difference error emerge?')
        fig.text(.02,.01,caption+' Fixed t0 cohort; dynamic trajectory regions; WM minus recorded effect.',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(root/'error_timeline.svg');plt.close(fig)
    confusion=load(root,'breakdown_confusion.csv')
    if confusion:
        fig,axes=plt.subplots(2,3,figsize=(10,6))
        for i,region in enumerate(('upstream','merge')):
            for j,arm in enumerate(('native','mappo','ours_c1')):
                r=next(r for r in confusion if r['region']==region and r['arm']==arm)
                mat=np.array([[int(r['TN']),int(r['FP'])],[int(r['FN']),int(r['TP'])]])
                ax=axes[i,j];ax.imshow(mat,cmap='Blues',vmin=0,vmax=36)
                for (a,b),v in np.ndenumerate(mat):ax.text(b,a,str(v),ha='center',va='center',color='black' if v<23 else 'white',fontsize=15)
                ax.set_xticks([0,1],['No event','Event']);ax.set_yticks([0,1],['No event','Event'])
                ax.set_xlabel('WM ensemble majority');ax.set_ylabel('Recorded event');ax.set_title(arm+' · '+region)
        fig.suptitle('Mainline speed-loss change points: real versus WM')
        fig.text(.02,.01,caption+' BIC step-vs-trend detector; no speed threshold; no event includes pre-existing congestion.',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(root/'breakdown_confusion.svg');plt.close(fig)
    signal=load(root,'paired_signal_summary.csv')
    if signal:
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,arm in zip(axes,('mappo','ours_c1')):
            rs=[r for r in signal if r['arm']==arm]
            for key,color,label in [('real_rms_dv_mps','#333333','Recorded'),('wm_rms_dv_mps','#275d89','WM2 K8 mean')]:
                good=[r for r in rs if r.get(key)]
                ax.plot([int(r['bin_end_s']) for r in good],[float(r[key]) for r in good],color=color,label=label)
            ax.set_title(arm+' − native');ax.set_xlabel('Time since warmup end (s)')
            ax.set_ylabel('HDV paired-speed RMS (m/s)');ax.legend(frameon=False)
        fig.suptitle('Growth / decay of controller-induced paired speed signals')
        fig.text(.02,.01,caption+' Common-present t0 HDVs; mean of world RMS, not an isolated-pulse stability test.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(root/'paired_signal.svg');plt.close(fig)
    pulse_path=root/'pulse_amplitude_worlds.csv'
    if pulse_path.exists():
        pulse=load(root,pulse_path.name)
        eligible=[r for r in pulse if int(r['target_lane'])<2]
        if eligible:
            fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
            for ax,name in zip(axes,('pulse_m','pulse_p')):
                for src,color,label in [('real','#333333','Recorded pulse'),('wm','#275d89','WM2 K8 mean')]:
                    rs=[r for r in eligible if r['pulse'].startswith(name) and r['source']==src and r['mean_amplitude_vs_target']]
                    ranks=sorted({int(r['rank']) for r in rs});ys=[]
                    for rank in ranks:
                        grouped={}
                        for r in rs:
                            if int(r['rank'])==rank:grouped.setdefault(r['seed'],[]).append(float(r['mean_amplitude_vs_target']))
                        ys.append(np.mean([np.mean(v) for v in grouped.values()]))
                    if ranks:ax.plot(ranks,ys,color=color,label=label)
                ax.axhline(1,color='grey',lw=.7,ls='--');ax.set_title(name+' · initial mainline platoon')
                ax.set_xlabel('Initial following rank (0 = pulse target)');ax.set_ylabel('Peak |paired Δv| / target amplitude (unitless)')
                ax.legend(frameon=False)
            fig.suptitle('Supplemental pulse propagation — outside E3 verdict worlds')
            fig.text(.02,.01,'Source: recorded qualification, seeds 969824–969835; WM2 K8; 0–30 s. Mean across worlds; support varies with rank; lane residence retained.',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(root/'pulse_gain_supplemental.svg');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();main(a.root)

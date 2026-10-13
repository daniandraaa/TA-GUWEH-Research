"""Static figures from independently verified V10 tables, no model inference."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.artifacts import ROOT,read_json,sha256_file,write_json
from research_v10.common import DEFAULT_RUN
from .audit_completed import DESTINATION


def plot():
    destination=ROOT/'results/research_v10'/DEFAULT_RUN/DESTINATION
    audit=read_json(destination/'audit.json')
    if audit['cases_verified']!=130 or not audit['all_saved_metrics_reproduced']:raise RuntimeError('Require completed audit')
    cases=pd.read_csv(destination/'cases_verified.csv');outer=cases[~cases.inner_selection]
    seed=pd.read_csv(destination/'seed_summary.csv')
    colors={'random_probe':'#5874a2','pretrained_probe':'#c66d38','pretrained_tail':'#7b5599'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for i,arm in enumerate(['random_probe','pretrained_probe']):
        x=outer[outer.arm.eq(arm)]
        for head_seed,offset in zip([42,43,44],[-.12,0,.12]):
            group=x[x.seed.eq(head_seed)]
            axes[0].scatter(np.full(len(group),i+offset),group.f1,color=colors[arm],alpha=.65,s=35)
        axes[0].errorbar(i,x.f1.mean(),yerr=x.f1.std(),fmt='D',color='black',capsize=5)
        axes[0].annotate(f'{x.f1.mean():.4f}',(i,x.f1.mean()),xytext=(8,8),textcoords='offset points')
    axes[0].set_xticks([0,1],['Random frozen','Pretrained frozen']);axes[0].set_ylim(0,1);axes[0].set_ylabel('Patient macro-F1')
    axes[0].set_title('Five folds × three fixed head seeds\nDiamond: mean; bar: descriptive SD')
    labels=['Random frozen','Pretrained frozen','Last-layer fine-tuning']
    pilot=outer[outer.fold.eq(1)&outer.seed.eq(42)]
    values=[float(pilot[pilot.arm.eq(arm)].f1.iloc[0]) for arm in ['random_probe','pretrained_probe','pretrained_tail']]
    bars=axes[1].bar(np.arange(3),values,color=list(colors.values()))
    axes[1].bar_label(bars,fmt='%.4f',padding=3);axes[1].set_xticks(np.arange(3),labels,rotation=18);axes[1].set_ylim(0,1)
    axes[1].set_title('Pilot only: fold 1, seed 42\n46 identical outer patients')
    fig.savefig(destination/'verified_probe_and_tail_pilot.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for arm in ['random_probe','pretrained_probe']:
        group=seed[seed.arm.eq(arm)]
        axes[0].plot(group.seed,group.mean_fold_macro_f1,marker='o',color=colors[arm],label=arm)
        fold=outer[outer.arm.eq(arm)].groupby('fold').f1.mean()
        axes[1].plot(fold.index,fold.values,marker='o',color=colors[arm],label=arm)
    for ax in axes:ax.set_ylim(0,1);ax.set_ylabel('Mean fold patient macro-F1');ax.grid(alpha=.2)
    axes[0].set_xticks([42,43,44]);axes[0].set_xlabel('Head seed');axes[0].set_title('Mean across five folds per seed')
    axes[1].set_xticks([1,2,3,4,5]);axes[1].set_xlabel('Outer fold');axes[1].set_title('Mean across three head seeds per fold')
    axes[0].legend();fig.savefig(destination/'verified_seed_and_fold_comparison.png',dpi=200);plt.close(fig)
    confusion=pd.read_csv(destination/'confusion_by_seed.csv');classes=['LVOT','RVOT_cusp','RVOT_septal_freewall_other']
    labels=['LVOT','RVOT cusp','RVOT other']
    fig,axes=plt.subplots(2,3,figsize=(11,7.4),layout='constrained')
    for row,arm in enumerate(['random_probe','pretrained_probe']):
        for col,head_seed in enumerate([42,43,44]):
            subset=confusion[confusion.arm.eq(arm)&confusion.seed.eq(head_seed)]
            cm=subset.pivot(index='true_class',columns='predicted_class',values='patients').loc[classes,classes].to_numpy()
            normalized=cm/cm.sum(1,keepdims=True)
            ax=axes[row,col];ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            for y in range(3):
                for x in range(3):ax.text(x,y,f'{cm[y,x]}\n{normalized[y,x]:.0%}',ha='center',va='center',color='white' if normalized[y,x]>.5 else 'black')
            ax.set_xticks(range(3),labels,rotation=20);ax.set_yticks(range(3),labels)
            ax.set_title(f'{arm}, head seed {head_seed}');ax.set_xlabel('Predicted');ax.set_ylabel('True')
    fig.suptitle('OOF confusion: 230 unique patients per panel; row percentages')
    fig.savefig(destination/'verified_oof_confusion.png',dpi=200);plt.close(fig)
    files={p.name:sha256_file(p) for p in sorted(destination.glob('verified_*.png'))}
    write_json(destination/'figure_provenance.json',{'audit_sha256':sha256_file(destination/'audit.json'),
        'plot_code_sha256':sha256_file(Path(__file__)),'files':files,'new_model_inference':False,'new_training':False})
    print('Verified figures:',list(files))


if __name__=='__main__':plot()

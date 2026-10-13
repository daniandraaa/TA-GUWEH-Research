"""Development-only engineering preview; no clinical beat annotation produced."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.artifacts import ROOT, read_json, write_json, sha256_file
from src.data import LEADS
from src.reports import plotting
from research_v2.common import context, DEFAULT_RUN
from research_v3.candidates import detect_candidates


def preview(count=12, seed=20261001):
    run,_,_,config,_=context(DEFAULT_RUN)
    records=[r for r in read_json(run/'data/patients.json') if r['pool']=='development']
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(records),min(count,len(records)),replace=False)
    spec=yaml.safe_load((ROOT/'research_v3/protocol.yaml').read_text())['candidate']
    out=run/'development_candidate_preview_20261001'
    out.mkdir(parents=True,exist_ok=True)
    plt=plotting(run)
    rows=[]
    for i in indices:
        r=records[int(i)]
        path=ROOT/'PVCVTRawECGData'/f'{r["hospital_id"]}.csv'
        raw=pd.read_csv(path,usecols=LEADS)[LEADS].to_numpy()
        peaks,qa=detect_candidates(raw,spec)
        rows.append({'hospital_id':r['hospital_id'],'class_name':r['class_name'],
                     'sublocation':r['sublocation'],'raw_sha256':sha256_file(path),**qa})
        stop=min(len(raw),5*spec['fs'])
        fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True)
        for ax,lead in zip(axes,[qa['lead_a'],qa['lead_b']]):
            channel=LEADS.index(lead)
            ax.plot(np.arange(stop)/spec['fs'],raw[:stop,channel],lw=.65,color='#243b53')
            points=peaks[peaks<stop]
            ax.scatter(points/spec['fs'],raw[points,channel],s=20,color='#d64545',label='Unverified consensus candidate')
            ax.set_ylabel(lead); ax.legend(loc='upper right',fontsize=8)
        axes[0].set_title(f'Patient {r["hospital_id"]}: engineering detector preview; PVC status unknown')
        axes[-1].set_xlabel('Time (seconds; 2000 Hz)')
        fig.tight_layout();fig.savefig(out/f'patient_{r["hospital_id"]}.png',dpi=150);plt.close(fig)
        print(f'PREVIEW {r["hospital_id"]}: {len(peaks)} retained candidates, agreement {qa["agreement"]:.3f}',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'engineering_quality.csv',index=False)
    write_json(out/'manifest.json',{'pool':'development','patient_count':len(frame),'selection':'seeded patient-uniform sample',
                'seed':seed,'detector_spec':spec,'detector_sha256':sha256_file(ROOT/'research_v3/candidates.py'),
                'pvc_verified':False,'expert_reviewed':False})
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients',type=int,default=12)
    args=parser.parse_args()
    print(preview(args.patients))

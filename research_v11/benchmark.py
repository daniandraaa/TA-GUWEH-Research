"""User-run LUDB 1.0.1 benchmark; gate is engineering, not clinical certification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from urllib.request import urlopen
import numpy as np
import pandas as pd
from src.artifacts import ROOT, read_json, sha256_file, write_json, canonical_hash
from research_v3.candidates import consensus_matches
from .common import LEADS, complete, verified, disk_check
from .morphology import filter_signal, boundary_envelope, delineate, candidates

BASE='https://physionet.org/files/ludb/1.0.1/'
EXTENSIONS=['hea','dat',*[lead.lower() for lead in LEADS]]
BEATS={'N','L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q'}


def wfdb_module():
    # Reuse the isolated QA dependency target from notebook 17; never install.
    import numpy, pandas, scipy
    if (ROOT/'.ecgqa').is_dir() and str(ROOT/'.ecgqa') not in sys.path:sys.path.insert(0,str(ROOT/'.ecgqa'))
    try:import wfdb
    except ImportError as error:raise RuntimeError('WFDB tidak tersedia. Gunakan target QA .ecgqa yang sudah dipakai notebook 17; jangan mengubah .venv training.') from error
    return wfdb


def annotation_qrs(samples,symbols):
    rows=[]
    for i,(sample,symbol) in enumerate(zip(samples,symbols)):
        if symbol in BEATS:
            if i==0 or i+1==len(samples) or symbols[i-1]!='(' or symbols[i+1]!=')':
                raise RuntimeError('LUDB QRS must have adjacent onset/peak/offset triplet')
            onset,offset=int(samples[i-1]),int(samples[i+1]);peak=int(sample)
            if not onset<peak<offset:raise RuntimeError('Invalid annotated boundary order')
            rows.append((onset,peak,offset))
    if not rows:raise RuntimeError('No annotated QRS in lead')
    return rows


def checksums(text):
    output={}
    for line in text.splitlines():
        parts=line.split(maxsplit=1)
        if len(parts)==2 and len(parts[0])==64:
            name=parts[1].lstrip('*').removeprefix('./');output[name]=parts[0]
    if not output:raise RuntimeError('No published LUDB checksums parsed')
    return output


def assets(study,download):
    folder=ROOT/'external_data/ludb_1.0.1';checksum=folder/'SHA256SUMS.txt'
    if not checksum.exists():
        if not download:raise RuntimeError('LUDB belum tersedia. Jalankan notebook 58 dengan DOWNLOAD_LUDB=True.')
        disk_check(study,64*1024**2);folder.mkdir(parents=True,exist_ok=True)
        with urlopen(BASE+'SHA256SUMS.txt',timeout=60) as response:text=response.read().decode('utf-8')
        checksums(text);checksum.write_text(text)
    hashes=checksums(checksum.read_text())
    def one(record):
        row={}
        for ext in EXTENSIONS:
            name=f'data/{record}.{ext}';target=folder/name
            if name not in hashes:raise RuntimeError(f'Published LUDB checksum missing: {name}')
            if not target.exists():
                if not download:raise RuntimeError(f'LUDB missing {name}; resume manual download')
                target.parent.mkdir(parents=True,exist_ok=True)
                with urlopen(BASE+name,timeout=60) as response:value=response.read()
                import hashlib
                if hashlib.sha256(value).hexdigest()!=hashes[name]:raise RuntimeError(f'LUDB download digest mismatch: {name}')
                temporary=target.with_name(target.name+'.tmp');temporary.write_bytes(value);temporary.replace(target)
            if sha256_file(target)!=hashes[name]:raise RuntimeError(f'Preserve changed LUDB file: {name}; do not silently overwrite')
            row[name]=hashes[name]
        return row
    output={}
    if download:
        disk_check(study,64*1024**2)
        with ThreadPoolExecutor(max_workers=8) as executor:
            pending=[executor.submit(one,i) for i in range(1,201)]
            for count,future in enumerate(as_completed(pending),1):
                output.update(future.result())
                if count%10==0:print(f'LUDB download/hash {count}/200 records',flush=True)
    else:
        for i in range(1,201):output.update(one(i))
    for name in ['LICENSE.txt','README','ANNOTATORS','RECORDS']:
        target=folder/name
        if name not in hashes:raise RuntimeError(f'LUDB metadata checksum absent: {name}')
        if not target.exists():
            if not download:raise RuntimeError(f'LUDB metadata missing: {name}; resume manual download')
            with urlopen(BASE+name,timeout=60) as response:value=response.read()
            if name not in hashes or __import__('hashlib').sha256(value).hexdigest()!=hashes[name]:raise RuntimeError('LUDB metadata checksum mismatch')
            target.write_bytes(value)
        if sha256_file(target)!=hashes[name]:raise RuntimeError(f'LUDB metadata changed: {name}')
        output[name]=hashes[name]
    return folder,output,sha256_file(checksum)


def evaluate_lead(envelope,reference,detected,fs,p):
    tolerance=round(fs*p['ludb_detection_tolerance_seconds'])
    lo,hi=reference[0][0]-tolerance,reference[-1][2]+tolerance
    current=np.array([x for x in detected if lo<=x<=hi],dtype=int)
    peaks=np.array([x[1] for x in reference],dtype=int)
    pairs=consensus_matches(peaks,current,tolerance)
    paired={r:c for r,c in pairs};errors=[]
    for onset,peak,offset in reference:
        for mode,center in [('reference_anchor',peak),('detected_anchor',paired.get(peak))]:
            bounds=delineate(envelope,int(center),fs,p) if center is not None else None
            row={'mode':mode,'reference_onset':onset,'reference_peak':peak,'reference_offset':offset,
                'estimated_onset':None if bounds is None else bounds[0],'estimated_offset':None if bounds is None else bounds[1],
                'valid_boundary':bounds is not None}
            if bounds is not None:
                row.update(onset_error_ms=abs(bounds[0]-onset)*1000/fs,offset_error_ms=abs(bounds[1]-offset)*1000/fs,
                    duration_error_ms=abs((bounds[1]-bounds[0])-(offset-onset))*1000/fs)
            errors.append(row)
    tp=len(pairs);fp=len(current)-tp;fn=len(peaks)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'reference_qrs':len(reference),'annotation_scoring_region':[int(lo),int(hi)]},errors


def gate_for(summary,p):
    g=p['ludb_gate'];detected=summary['boundaries']['detected_anchor'];oracle=summary['boundaries']['reference_anchor']
    checks={'detection_f1':summary['micro_detection_f1']>=g['minimum_detection_f1']}
    for mode,values in [('reference',oracle),('detected',detected)]:
        checks[mode+'_coverage']=values['coverage']>=g['minimum_boundary_coverage']
        for metric in ['onset','offset','duration']:
            value=values[f'median_{metric}_error_ms']
            checks[f'{mode}_{metric}']=value is not None and value<=g[f'maximum_median_{metric}_error_ms']
    return checks,bool(all(checks.values()))


def run(study,download=False):
    wfdb=wfdb_module();folder,hashes,checksum=assets(study,download)
    destination=study.run/'ludb';destination.mkdir(parents=True,exist_ok=True)
    identity={'experiment_hash':study.signature,'dataset':'LUDB 1.0.1','wfdb_version':wfdb.__version__,
        'published_checksums_sha256':checksum,'assets':hashes,'records':list(range(1,201)),
        'scope':'delineation_not_pvc_site_validation','threshold_tuning_on_ludb':False}
    old=verified(destination,identity)
    if old:
        result=read_json(destination/'summary.json');print('LUDB verified; PASS=',result['passed'],flush=True);return result
    disk_check(study,24*1024**2);rows=[];errors=[]
    write_json(destination/'identity.json',identity)
    for record in range(1,201):
        shard=destination/f'record_{record}.json'
        if shard.exists():
            saved=read_json(shard)
            if saved['identity_hash']!=canonical_hash(identity):raise RuntimeError('LUDB resume identity changed')
            rows.extend(saved['scores']);errors.extend(saved['errors']);continue
        name=str(folder/f'data/{record}');data=wfdb.rdrecord(name)
        lookup={lead.upper():i for i,lead in enumerate(data.sig_name)}
        if any(lead.upper() not in lookup for lead in LEADS):raise RuntimeError('LUDB lead mismatch')
        raw=data.p_signal[:,[lookup[lead.upper()] for lead in LEADS]];fs=float(data.fs)
        envelope=boundary_envelope(raw,fs,study.protocol)
        centers,_=candidates(raw,fs);record_rows=[];record_errors=[]
        for j,lead in enumerate(LEADS):
            ann=wfdb.rdann(name,lead.lower());reference=annotation_qrs(ann.sample,ann.symbol)
            scores,measurements=evaluate_lead(envelope[:,j],reference,centers,fs,study.protocol)
            record_rows.append({'record':record,'lead':lead,**scores})
            record_errors.extend({'record':record,'lead':lead,**row} for row in measurements)
        write_json(shard,{'identity_hash':canonical_hash(identity),'scores':record_rows,'errors':record_errors})
        rows.extend(record_rows);errors.extend(record_errors);print(f'LUDB benchmark {record}/200',flush=True)
    scores=pd.DataFrame(rows);bounds=pd.DataFrame(errors);scores.to_csv(destination/'detection.csv',index=False);bounds.to_csv(destination/'boundary_errors.csv',index=False)
    tp,fp,fn=map(int,[scores[k].sum() for k in ['tp','fp','fn']])
    summary={'records':200,'lead_evaluations':2400,'micro_detection_f1':2*tp/(2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,'boundaries':{}}
    for mode in ['reference_anchor','detected_anchor']:
        sub=bounds[bounds['mode']==mode];good=sub[sub.valid_boundary]
        summary['boundaries'][mode]={'reference_qrs':len(sub),'valid_boundaries':len(good),'coverage':len(good)/len(sub),
            **{f'{stat}_{metric}_error_ms':(float(good[f'{metric}_error_ms'].median()) if stat=='median' else float(good[f'{metric}_error_ms'].quantile(.95))) if len(good) else None
                for stat in ['median','p95'] for metric in ['onset','offset','duration']}}
    checks,passed=gate_for(summary,study.protocol)
    summary.update(passed=passed,checks=checks,gate_scope='prespecified_engineering_gate_not_clinical_certification',
        repeated_beats_across_leads_not_independent_patients=True,pvc_origin_validation=False,training_performed=False)
    write_json(destination/'summary.json',summary)
    names=['identity.json','detection.csv','boundary_errors.csv','summary.json',*[f'record_{i}.json' for i in range(1,201)]]
    complete(destination,identity,names,passed=passed)
    print('LUDB gate:',passed,checks,flush=True);return summary


def require_pass(study):
    folder=study.run/'ludb'
    if not (folder/'completed.json').exists():raise RuntimeError('Jalankan notebook 58 / LUDB dahulu')
    identity=read_json(folder/'identity.json');verified(folder,identity)
    if identity['experiment_hash']!=study.signature:raise RuntimeError('LUDB belongs to different protocol')
    asset_folder=ROOT/'external_data/ludb_1.0.1'
    for name,digest in identity['assets'].items():
        if sha256_file(asset_folder/name)!=digest:raise RuntimeError('LUDB source changed')
    result=read_json(folder/'summary.json');checks,passed=gate_for(result,study.protocol)
    if checks!=result['checks'] or passed!=result['passed']:raise RuntimeError('LUDB gate result changed')
    if not passed:raise RuntimeError('LUDB boundary/detection gate FAIL. Pertahankan hasil, periksa diagnosis; jangan melatih arm morfologi. Baseline random_l2/pretrained_l2 boleh dijalankan terpisah.')
    return sha256_file(folder/'completed.json')

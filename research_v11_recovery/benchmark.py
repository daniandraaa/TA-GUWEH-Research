"""Decode incomplete LUDB annotations without inventing reference boundaries."""
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.artifacts import write_json,sha256_file
from research_v3.candidates import consensus_matches
from research_v11 import benchmark as frozen
from research_v11.common import complete


def annotation_qrs(samples,symbols):
    if len(samples)!=len(symbols):raise RuntimeError('Annotation sample/symbol lengths differ')
    rows=[]
    for i,(sample,symbol) in enumerate(zip(samples,symbols)):
        if symbol not in frozen.BEATS:continue
        # Use only directly adjacent labels. Never borrow a P/T boundary,
        # substitute a peak as onset, or reorder a contradictory annotation.
        onset=int(samples[i-1]) if i and symbols[i-1]=='(' else None
        offset=int(samples[i+1]) if i+1<len(samples) and symbols[i+1]==')' else None
        rows.append((onset,int(sample),offset))
    if not rows:raise RuntimeError('No annotated QRS peaks in lead')
    if any(a[1]>=b[1] for a,b in zip(rows,rows[1:])):raise RuntimeError('Annotated QRS peaks not strictly increasing')
    return rows


def reference_valid(row):
    onset,peak,offset=row
    return onset is not None and offset is not None and onset<peak<offset


def evaluate_lead(envelope,reference,detected,fs,p):
    tolerance=round(fs*p['ludb_detection_tolerance_seconds'])
    first,last=reference[0],reference[-1]
    lo=min(first[1],first[0] if first[0] is not None else first[1])-tolerance
    hi=max(last[1],last[2] if last[2] is not None else last[1])+tolerance
    current=np.array([x for x in detected if lo<=x<=hi],dtype=int)
    peaks=np.array([row[1] for row in reference],dtype=int)
    pairs=consensus_matches(peaks,current,tolerance);paired={r:c for r,c in pairs};errors=[]
    for reference_row in reference:
        onset,peak,offset=reference_row;available=reference_valid(reference_row)
        reason=None if available else ('missing_reference_boundary' if onset is None or offset is None else 'inconsistent_reference_boundary_order')
        for mode,center in [('reference_anchor',peak),('detected_anchor',paired.get(peak))]:
            bounds=frozen.delineate(envelope,int(center),fs,p) if center is not None else None
            row={'mode':mode,'reference_onset':onset,'reference_peak':peak,'reference_offset':offset,
                'reference_boundary_available':available,'reference_boundary_issue':reason,
                'estimated_onset':None if bounds is None else bounds[0],'estimated_offset':None if bounds is None else bounds[1],
                'estimated_boundary_available':bounds is not None,'valid_boundary':bool(available and bounds is not None)}
            if row['valid_boundary']:
                row.update(onset_error_ms=abs(bounds[0]-onset)*1000/fs,offset_error_ms=abs(bounds[1]-offset)*1000/fs,
                           duration_error_ms=abs((bounds[1]-bounds[0])-(offset-onset))*1000/fs)
            errors.append(row)
    tp=len(pairs);fp=len(current)-tp;fn=len(peaks)-tp
    return {'tp':tp,'fp':fp,'fn':fn,'reference_qrs':len(reference),
        'reference_complete_qrs':sum(reference_valid(row) for row in reference),'annotation_scoring_region':[int(lo),int(hi)]},errors


def run(study,download=False):
    with patch.object(frozen,'annotation_qrs',annotation_qrs),patch.object(frozen,'evaluate_lead',evaluate_lead):
        result=frozen.run(study,download)
    folder=study.run/'ludb';bounds=pd.read_csv(folder/'boundary_errors.csv')
    reference=bounds[bounds['mode'].eq('reference_anchor')]
    if 'reference_boundary_available' not in reference:raise RuntimeError('Require recovery annotation schema')
    available=reference.reference_boundary_available.astype(bool)
    report={'annotated_qrs_retained':len(reference),'complete_reference_qrs':int(available.sum()),
        'unavailable_reference_qrs':int((~available).sum()),'reference_annotation_coverage':float(available.mean()),
        'issues':reference.reference_boundary_issue.dropna().value_counts().to_dict(),
        'unavailable_reference_qrs_retained_in_boundary_coverage_denominator':True,
        'reference_boundaries_invented':False,'gate_thresholds_changed':False,'new_training_performed':False,
        'experiment_hash':study.signature,'benchmark_manifest_sha256':sha256_file(folder/'completed.json')}
    audit_dir=folder/'annotation_audit';audit_dir.mkdir(parents=True,exist_ok=True)
    write_json(audit_dir/'annotation_integrity.json',report)
    complete(audit_dir,{'experiment_hash':study.signature,'benchmark_manifest_sha256':report['benchmark_manifest_sha256']},
        ['annotation_integrity.json'])
    print('Reference annotation integrity:',report,flush=True)
    return result

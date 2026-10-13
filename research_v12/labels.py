"""Match source annotation symbols, without inferring a clinical label."""
import numpy as np
from research_v3.candidates import consensus_matches

BEATS = {'N','L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q'}


def beat_class(symbol):
    if symbol=='N':
        return 0
    if symbol in {'V','r'}:
        return 1
    if symbol in BEATS-{'N','V','r','Q'}:
        return 2
    return -1  # Q/unclassifiable and non-beat markers never become normal.


def annotation_rows(samples, symbols, length, fs, p):
    if len(samples)!=len(symbols):
        raise ValueError('Annotation lengths differ')
    before,after = round(fs*p['crop_before_seconds']),round(fs*p['crop_after_seconds'])
    rows=[];excluded=0
    for center,symbol in zip(samples,symbols):
        if symbol not in BEATS:
            continue
        center=int(center)
        if center<before or center+after>length:
            excluded+=1
            continue
        rows.append((center,beat_class(symbol),symbol))
    if any(a[0]>=b[0] for a,b in zip(rows,rows[1:])):
        raise ValueError('Eligible reference beats must be strictly increasing')
    return rows,excluded


def match_labels(reference, centers, fs, tolerance):
    reference_centers=np.array([r[0] for r in reference],dtype=int)
    pairs=consensus_matches(reference_centers,centers,round(fs*tolerance))
    lookup={r[0]:r for r in reference}
    match={c:lookup[r] for r,c in pairs}
    # -2 unmatched detector event; -1 matched unknown reference.
    labels=np.array([match[int(c)][1] if int(c) in match else -2 for c in centers],dtype=int)
    matched=np.array([match[int(c)][0] if int(c) in match else -1 for c in centers],dtype=int)
    return labels,matched


def pvc_counts(reference, labels, pvc_decisions):
    labels=np.asarray(labels);decisions=np.asarray(pvc_decisions,dtype=bool)
    if labels.shape!=decisions.shape:
        raise ValueError('Prediction/label lengths differ')
    n_positive=sum(r[1]==1 for r in reference)
    tp=int(np.sum(decisions & (labels==1)))
    # Ignore unknown reference Q; count unmatched PVC detections as false alarms.
    fp=int(np.sum(decisions & (labels!=-1) & (labels!=1)))
    fn=n_positive-tp
    if fn<0:
        raise ValueError('A reference must not match multiple predictions')
    return {'tp':tp,'fp':fp,'fn':fn,'positive_reference':n_positive,
        'unknown_reference_predictions':int(np.sum(labels==-1)),
        'PVC_predictions_on_unmatched_candidates':int(np.sum(decisions & (labels==-2)))}


def count_metrics(counts):
    tp,fp,fn = (int(counts[k]) for k in ['tp','fp','fn'])
    if min(tp,fp,fn)<0:
        raise ValueError('Negative count')
    return {'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

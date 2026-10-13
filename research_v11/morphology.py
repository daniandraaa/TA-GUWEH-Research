"""Label-free candidate measurements; boundaries are estimates until benchmarked."""
import copy
import numpy as np
from scipy import signal
import yaml
from src.artifacts import ROOT
from research_v3.candidates import detect_candidates
from .common import LEADS

RR_NAMES=['rr_before_ms','rr_after_ms','rr_before_after_ratio']
GLOBAL_NAMES=['robust_qrs_span_ms','first_r_dominant_v_index','v2_s_over_v3_r']
LEAD_METRICS=['width_ms','r_relative','s_relative','r_fraction','signed_area_relative_seconds']
BASE_NAMES=[f'{lead}_{metric}' for lead in LEADS for metric in LEAD_METRICS]+GLOBAL_NAMES+RR_NAMES


def filter_signal(raw,fs,band):
    values=np.asarray(raw,dtype=np.float64)
    if values.ndim not in {1,2} or not np.isfinite(values).all() or len(values)<int(fs):
        raise ValueError('Finite continuous signal of at least one second required')
    sos=signal.butter(3,[band[0],min(band[1],fs*.45)],btype='bandpass',fs=fs,output='sos')
    return signal.sosfiltfilt(sos,values,axis=0)


def candidates(raw,fs):
    spec=copy.deepcopy(yaml.safe_load((ROOT/'research_v3/protocol.yaml').read_text())['candidate'])
    spec.update(fs=fs,before_seconds=0.,after_seconds=0.,maximum_per_patient=len(raw))
    return detect_candidates(raw,spec)


def boundary_envelope(raw,fs,p):
    # Separate boundary branch, not the high-pass candidate detector branch.
    x=filter_signal(raw,fs,p['boundary_filter_hz'])
    d=np.abs(np.gradient(x,axis=0))
    width=max(1,round(fs*p['boundary_smoothing_seconds']))
    if d.ndim==1:return np.convolve(d,np.ones(width)/width,mode='same')
    return np.column_stack([np.convolve(d[:,i],np.ones(width)/width,mode='same') for i in range(d.shape[1])])


def delineate(envelope,center,fs,p):
    """Find sustained quiet runs on both sides of candidate derivative activity."""
    e=np.asarray(envelope,dtype=float);radius=round(fs*p['boundary_search_seconds']);quiet=max(1,round(fs*p['boundary_quiet_seconds']))
    lo,hi=max(0,center-radius),min(len(e),center+radius+1)
    if lo==0 or hi==len(e):return None
    around=e[max(lo,center-round(.05*fs)):min(hi,center+round(.05*fs)+1)]
    peak=float(around.max());noise=float(np.quantile(e[lo:hi],.20))
    if peak<=np.finfo(float).eps:return None
    threshold=max(p['boundary_relative_threshold']*peak,2*noise)
    active=e[lo:hi]>threshold
    pivot_lo=max(lo,center-round(.05*fs));pivot_hi=min(hi,center+round(.05*fs)+1)
    # Bracket all activity in the central core: a quiet valley between two
    # QRS slopes must not be mistaken for the end of a biphasic complex.
    core=np.flatnonzero(active[pivot_lo-lo:pivot_hi-lo])
    if not len(core):return None
    left_pivot=pivot_lo-lo+int(core[0]);right_pivot=pivot_lo-lo+int(core[-1])
    left=right=None
    for stop in range(left_pivot,quiet-1,-1):
        if not active[stop-quiet:stop].any():left=lo+stop;break
    for start in range(right_pivot+1,len(active)-quiet+1):
        if not active[start:start+quiet].any():right=lo+start;break
    if left is None or right is None:return None
    pad=round(fs*p['boundary_padding_seconds']);left-=pad;right+=pad
    duration=(right-left)*1000/fs
    if not left<center<right or not p['boundary_duration_ms'][0]<=duration<=p['boundary_duration_ms'][1]:return None
    return int(left),int(right)


def summarize(values,names):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(names):raise ValueError('Feature schema mismatch')
    row={}
    for j,name in enumerate(names):
        good=a[:,j][np.isfinite(a[:,j])]
        row[name+'_median']=float(np.median(good)) if len(good) else np.nan
        row[name+'_iqr']=float(np.quantile(good,.75)-np.quantile(good,.25)) if len(good) else np.nan
    return row


def measure_record(raw,fs,p):
    if np.asarray(raw).ndim!=2 or np.asarray(raw).shape[1]!=12:raise ValueError('Twelve fixed-order leads required')
    clean=filter_signal(raw,fs,p['morphology_filter_hz']);peaks,qa=candidates(raw,fs)
    envelope=boundary_envelope(raw,fs,p);scale=float(np.sqrt(np.mean(clean**2)))
    if scale<=1e-12:scale=1.
    rows=[];before=round(fs*p['complete_crop_before_seconds']);after=round(fs*p['complete_crop_after_seconds'])
    for index,center in enumerate(peaks):
        if center<before or center+after>=len(clean):continue
        values={name:np.nan for name in BASE_NAMES};onsets=[];offsets=[];rs={};boundary_fields={}
        for j,lead in enumerate(LEADS):
            boundary_fields.update({f'{lead}_onset_sample':None,f'{lead}_offset_sample':None})
            bounds=delineate(envelope[:,j],int(center),fs,p)
            if bounds is None:continue
            onset,offset=bounds;baseline=np.median(clean[max(0,onset-round(.04*fs)):onset,j])
            segment=clean[onset:offset+1,j]-baseline
            r=max(0.,float(segment.max()));s=max(0.,float(-segment.min()));width=(offset-onset)*1000/fs
            # One common per-record scale preserves relative amplitudes across leads.
            fields=[width,r/scale,s/scale,r/(r+s) if r+s>scale*.01 else np.nan,float(np.trapezoid(segment,dx=1/fs)/scale)]
            values.update({f'{lead}_{m}':v for m,v in zip(LEAD_METRICS,fields)})
            onsets.append(onset);offsets.append(offset);rs[lead]=(r,s)
            boundary_fields.update({f'{lead}_onset_sample':int(onset),f'{lead}_offset_sample':int(offset)})
        if len(onsets)>=6:values['robust_qrs_span_ms']=(np.quantile(offsets,.9)-np.quantile(onsets,.1))*1000/fs
        if all(lead in rs for lead in LEADS[6:]):
            positive=[i+1 for i,lead in enumerate(LEADS[6:]) if rs[lead][0]>=rs[lead][1] and sum(rs[lead])>scale*.01]
            values['first_r_dominant_v_index']=float(positive[0]) if positive else np.nan
        if 'V2' in rs and 'V3' in rs and rs['V3'][0]>scale*.01:values['v2_s_over_v3_r']=rs['V2'][1]/rs['V3'][0]
        if index:values['rr_before_ms']=(center-peaks[index-1])*1000/fs
        if index+1<len(peaks):values['rr_after_ms']=(peaks[index+1]-center)*1000/fs
        if np.isfinite(values['rr_before_ms']) and np.isfinite(values['rr_after_ms']):
            values['rr_before_after_ratio']=values['rr_before_ms']/values['rr_after_ms']
        rows.append({'center_sample':int(center),'valid_leads':len(onsets),**boundary_fields,**values})
    measurements=np.array([[row[n] for n in BASE_NAMES] for row in rows],dtype=float).reshape(-1,len(BASE_NAMES))
    summary=summarize(measurements,BASE_NAMES)
    quality={'total_qrs_candidates':len(peaks),'complete_candidates':len(rows),
        'candidates_with_at_least_six_valid_leads':sum(row['valid_leads']>=6 for row in rows),
        'feature_missing_fraction':float(np.mean(~np.isfinite(list(summary.values())))),
        'record_seconds':len(raw)/fs,'global_rms_native_units':scale,'detector_agreement':qa['agreement'],
        'pvc_verified':False,'clinical_qrs_duration_validated':False}
    return summary,quality,rows,clean


def columns_for(arm):
    if arm=='qrs_rr':names=[n for n in BASE_NAMES if n.endswith('_width_ms') or n=='robust_qrs_span_ms' or n in RR_NAMES]
    elif arm=='morphology':names=[n for n in BASE_NAMES if n not in RR_NAMES]
    elif arm in {'morphology_rr','fusion_random','fusion_pretrained'}:names=BASE_NAMES
    else:raise ValueError('Unknown morphology group')
    return [name+suffix for name in names for suffix in ('_median','_iqr')]

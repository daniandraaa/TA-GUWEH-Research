"""Lead-agnostic beat descriptors at detected centers, offline RR context."""
import numpy as np
from scipy import signal
from research_v3.candidates import lead_candidates, consensus_matches


def feature_names(p, rr=False):
    names=[f'shape_{i:02}' for i in range(p['feature_samples'])]
    names+=['r_relative','s_relative','r_fraction','signed_area_relative_s','derivative_rms_relative',
        'energy_width_10_90_ms','energy_centroid_ms','energy_spread_ms']
    if rr:
        names+=['rr_before_s','rr_after_s','rr_ratio']
    return names


def choose_leads(names, dataset, p):
    preferred=p['mit_preferred_feature_leads'] if dataset=='mitdb' else p['preferred_feature_leads']
    lookup={str(n).upper():i for i,n in enumerate(names)}
    indices=[lookup[n.upper()] for n in preferred if n.upper() in lookup]
    if not indices:
        # Actual channel names are retained; an alternative is never renamed II.
        indices=[0]
    return indices


def clean_signal(raw, fs, p):
    raw=np.asarray(raw,dtype=float)
    if raw.ndim!=2 or len(raw)<fs*2 or not np.isfinite(raw).all():
        raise ValueError('Finite continuous [samples,channels] >=2 seconds required')
    sos=signal.butter(3,p['feature_filter_hz'],fs=fs,btype='bandpass',output='sos')
    return signal.sosfiltfilt(sos,raw,axis=0)


def detect(raw, fs, p):
    """Generalize the frozen energy detector to the selected 1/2 channels."""
    sos=signal.butter(3,p['candidate']['bandpass_hz'],fs=fs,btype='bandpass',output='sos')
    filtered=signal.sosfiltfilt(sos,raw,axis=0)
    order=sorted(range(raw.shape[1]),key=lambda j:(-float(np.quantile(np.abs(filtered[:,j]),.95)),j))
    peaks=[lead_candidates(filtered[:,j],fs,p['candidate']) for j in order[:2]]
    if len(peaks)==2:
        centers=np.array([a for a,_ in consensus_matches(peaks[0],peaks[1],round(fs*p['candidate']['consensus_seconds']))],dtype=int)
        mode='two_lead_consensus'
    else:
        centers=peaks[0];mode='single_available_channel'
    before,after=round(fs*p['crop_before_seconds']),round(fs*p['crop_after_seconds'])
    centers=centers[(centers>=before)&(centers+after<=len(raw))]
    return centers,{'mode':mode,'channels':raw.shape[1],'candidate_count':len(centers)}


def single_features(crop, fs, scale, p):
    x=np.asarray(crop,dtype=float)
    baseline=float(np.median(np.r_[x[:max(1,round(.04*fs))],x[-max(1,round(.04*fs)):]]))
    x=(x-baseline)/max(float(scale),1e-8)
    # Fourier resampling applies low-pass truncation before the fixed grid.
    shape=signal.resample(x,p['feature_samples'])
    r=max(0.,float(x.max()));s=max(0.,float(-x.min()))
    d=np.gradient(x)*fs;energy=d*d;total=float(energy.sum())
    t=(np.arange(len(x))-round(fs*p['crop_before_seconds']))/fs
    if total>1e-12:
        distribution=np.cumsum(energy)/total
        lo=int(np.searchsorted(distribution,.1));hi=min(len(x)-1,int(np.searchsorted(distribution,.9)))
        centroid=float(np.sum(t*energy)/total)
        spread=float(np.sqrt(np.sum((t-centroid)**2*energy)/total))
        width=(hi-lo)*1000/fs
    else:
        width=centroid=spread=0.
    descriptors=[r,s,r/(r+s) if r+s>1e-10 else 0.5,float(np.trapezoid(x,dx=1/fs)),
        float(np.sqrt(np.mean(d*d))),width,centroid*1000,spread*1000]
    return np.r_[shape,descriptors]


def extract_array(raw, centers, fs, p):
    clean=clean_signal(raw,fs,p)
    before,after=round(fs*p['crop_before_seconds']),round(fs*p['crop_after_seconds'])
    scales=np.maximum(np.sqrt(np.mean(clean**2,axis=0)),1e-8)
    n=raw.shape[1];width=len(feature_names(p,True))
    x=np.empty((len(centers),n,width),dtype=np.float32)
    for i,center in enumerate(centers):
        if center<before or center+after>len(clean):
            raise ValueError('Incomplete crop must be counted/excluded explicitly')
        rr_before=(center-centers[i-1])/fs if i else np.nan
        rr_after=(centers[i+1]-center)/fs if i+1<len(centers) else np.nan
        ratio=rr_before/rr_after if np.isfinite(rr_before) and np.isfinite(rr_after) and rr_after>0 else np.nan
        for lead in range(n):
            x[i,lead]=np.r_[single_features(clean[center-before:center+after,lead],fs,scales[lead],p),rr_before,rr_after,ratio]
    if np.isinf(x).any():
        raise RuntimeError('Infinite feature')
    return x

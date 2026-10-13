"""Score all eligible reference PVCs, including beats missed by the detector."""
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score
from .labels import pvc_counts,count_metrics


def pvc_decisions(prob,threshold):
    prob=np.asarray(prob)
    if prob.ndim!=2 or prob.shape[1]!=3 or not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any() or not np.allclose(prob.sum(axis=1),1.,atol=1e-6):
        raise ValueError('Finite [candidates,3] probabilities summing to one required')
    return (prob[:,1]>=threshold)&(prob.argmax(axis=1)==1)


def evaluate(blocks,probabilities,threshold,normal_threshold=.9):
    if len(blocks)!=len(probabilities):
        raise ValueError('Block/probability lengths differ')
    per_record=[];ys=[];ps=[]
    normal_tp=normal_fp=normal_reference=normal_unknown=pvc_unknown=0
    for block,prob in zip(blocks,probabilities):
        prob=np.asarray(prob);labels=block['labels']
        if len(prob)!=len(labels):
            raise ValueError('Beat/prediction length differs')
        pvc=pvc_decisions(prob,threshold)
        counts=pvc_counts(block['reference'],labels,pvc)
        pvc_unknown+=int(np.sum(pvc&(labels==-1)))
        normal_reference+=sum(r[1]==0 for r in block['reference'])
        known=labels>=0
        if len(prob):
            n=(prob[:,0]>=normal_threshold)&(prob.argmax(axis=1)==0)
            normal_tp+=int(np.sum(n&(labels==0)))
            normal_fp+=int(np.sum(n&(labels!=0)&(labels!=-1)))
            normal_unknown+=int(np.sum(n&(labels==-1)))
        ys.append(labels[known]);ps.append(prob[known])
        per_record.append({'record':block['record'],'group':block['group'],**counts,
            'candidates':len(labels),'known_matched':int(known.sum()),'references':len(block['reference']),
            'missed_reference_PVC':sum(r[1]==1 for r in block['reference'])-int(np.sum(labels==1))})
    counts={k:sum(r[k] for r in per_record) for k in ['tp','fp','fn']}
    y=np.concatenate(ys);prob=np.concatenate(ps);n=len(y)
    if n:
        pred=prob.argmax(axis=1);binary=y==1
        diagnostic={'matched_known_macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'matched_known_confusion':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'matched_known_PVC_PR_AUC':float(average_precision_score(binary,prob[:,1])) if binary.any() else None,
            'matched_known_PVC_ROC_AUC':float(roc_auc_score(binary,prob[:,1])) if len(np.unique(binary))==2 else None,
            'matched_known_PVC_brier':float(np.mean((prob[:,1]-binary)**2))}
    else:
        diagnostic={'matched_known_macro_f1':0.,'matched_known_confusion':[[0]*3 for _ in range(3)],
            'matched_known_PVC_PR_AUC':None,'matched_known_PVC_ROC_AUC':None,'matched_known_PVC_brier':None}
    result={**counts,**count_metrics(counts),'normal_N_high_score_tp':normal_tp,'normal_N_high_score_fp':normal_fp,
        'normal_N_high_score_precision':normal_tp/(normal_tp+normal_fp) if normal_tp+normal_fp else 0.,
        'normal_N_reference':normal_reference,
        'normal_N_high_score_reference_recall':normal_tp/normal_reference if normal_reference else 0.,
        'normal_N_high_score_selected':normal_tp+normal_fp,'matched_known_candidates':n,
        'normal_N_high_score_all_selected':normal_tp+normal_fp+normal_unknown,
        'normal_N_predictions_on_unknown_Q':normal_unknown,'PVC_predictions_on_unknown_Q':pvc_unknown,
        'unknown_Q_candidates_ignored':sum(r['unknown_reference_predictions'] for r in per_record),
        'eligible_references':sum(r['references'] for r in per_record),
        'candidate_count':sum(r['candidates'] for r in per_record),
        'missed_reference_PVC':sum(r['missed_reference_PVC'] for r in per_record),
        'PVC_threshold':float(threshold),'normal_N_threshold':float(normal_threshold),**diagnostic}
    return result,per_record


def group_bootstrap(rows,repeats=2000,seed=20261007):
    groups=sorted({r['group'] for r in rows})
    if not groups or repeats<1:
        raise ValueError('Nonempty source groups and positive repeat count required')
    counts=np.array([[sum(r[k] for r in rows if r['group']==g) for k in ['tp','fp','fn']] for g in groups])
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repeats):
        a=counts[rng.integers(len(groups),size=len(groups))].sum(axis=0)
        draws.append(list(count_metrics(dict(zip(['tp','fp','fn'],a))).values()))
    q=np.quantile(draws,[.025,.975],axis=0)
    return {k:[float(q[0,i]),float(q[1,i])] for i,k in enumerate(['precision','recall','f1'])}

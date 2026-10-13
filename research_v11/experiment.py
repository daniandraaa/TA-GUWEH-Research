"""Manual deterministic L2 heads: inner selection, fresh outer refit, no test."""
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.artifacts import read_json, sha256_file, write_json
from src.evaluation import compute_metrics, prediction_table
from .common import FEATURE_ARMS, complete, verified, disk_check
from .morphology import columns_for
from .features import read_features
from .benchmark import require_pass


def build_pipeline(C,p):
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scaler',StandardScaler()),('classifier',LogisticRegression(C=float(C),l1_ratio=0.,solver='lbfgs',
            class_weight='balanced',max_iter=p['maximum_iterations'],tol=p['tolerance'],random_state=42))])


def fit_pipeline(values,labels,C,p):
    model=build_pipeline(C,p)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(np.asarray(values,dtype=float),np.asarray(labels,dtype=int))
    if not np.array_equal(model.classes_,[0,1,2]):raise RuntimeError('Three training classes required')
    return model


def select(rows):
    if len(rows)!=9 or len({row['C'] for row in rows})!=9:raise RuntimeError('All nine C scenarios required')
    return min(rows,key=lambda row:(-row['evaluation_f1'],row['C']))


def load_bundle(study,arms):
    if any(arm not in study.protocol['arms'] for arm in arms):raise ValueError('Unsupported arm')
    patients=pd.read_csv(study.feature_folder/'patients.csv',dtype={'hospital_id':str})
    arrays={arm:np.load(study.feature_folder/f'{arm}_probe.npy',mmap_mode='r') for arm in ['random','pretrained']}
    if any(a.shape!=(230,768) or not np.isfinite(a).all() for a in arrays.values()):raise RuntimeError('Invalid frozen features')
    matrices={};sources={'v10_feature_manifest':study.identity['feature_manifest_sha256']}
    morphology=None
    if any(arm in FEATURE_ARMS for arm in arms):
        sources['ludb_gate']=require_pass(study);morphology,sources['morphology_manifest']=read_features(study)
    for arm in arms:
        if arm in ['random_l2','pretrained_l2']:matrix=np.asarray(arrays[arm.split('_')[0]],dtype=float)
        else:
            matrix=morphology[columns_for(arm)].to_numpy(dtype=float)
            if arm.startswith('fusion_'):matrix=np.concatenate([np.asarray(arrays[arm.split('_')[1]],dtype=float),matrix],axis=1)
        if np.isinf(matrix).any():raise RuntimeError('Infinite features prohibited; missing measurements stay NaN for training-only imputation')
        matrices[arm]=matrix
    return patients,matrices,sources


def case(study,arm,fold,C,phase,patients,values,sources,choice=None):
    split=study.folds[fold-1];training=sorted(split['gan_fit_ids'] if phase=='inner' else split['gan_fit_ids']+split['gan_reference_ids'])
    evaluation=sorted(split['gan_reference_ids'] if phase=='inner' else split['outer_validation_ids'])
    identity={'experiment_hash':study.signature,'arm':arm,'fold':fold,'C':float(C),'phase':phase,
        'training_ids':training,'evaluation_ids':evaluation,'source_manifests':sources,
        'choice_manifest':None if phase=='inner' else sha256_file(choice),'preprocessing_fit_ids':training,
        'class_weight_fit_ids':training,'raw_feature_count':values.shape[1],
        'method':'regularized_multinomial_logistic_regression','outer_callbacks':False,'test_signals_read':False}
    scenario=study.protocol['inverse_regularization_C'].index(float(C))+1
    folder=study.run/'cases'/arm/f'fold_{fold}'/(f'inner_S{scenario:02}' if phase=='inner' else 'outer')
    if verified(folder,identity):print(f'SKIP {arm}/fold{fold}/{folder.name}',flush=True);return read_json(folder/'result.json')
    folder.mkdir(parents=True,exist_ok=True);disk_check(study,2*1024**2)
    if (folder/'identity.json').exists() and read_json(folder/'identity.json')!=identity:raise RuntimeError('Case resume identity changed')
    write_json(folder/'identity.json',identity);lookup={hid:i for i,hid in enumerate(patients.hospital_id)}
    train_index=[lookup[hid] for hid in training];eval_index=[lookup[hid] for hid in evaluation];labels=patients.label.to_numpy()
    print(f'FIT CPU {arm} fold{fold} {phase} C={C:g} patients={len(training)}/{len(evaluation)}',flush=True)
    try:model=fit_pipeline(values[train_index],labels[train_index],C,study.protocol)
    except ConvergenceWarning as error:
        write_json(folder/'failure.json',{'error':str(error),'converged':False,'scores_reported':False})
        raise RuntimeError(f'L2 solver belum converged: {folder}. Pertahankan kegagalan; jangan memilih skor dari fit ini.') from error
    probability=model.predict_proba(values[eval_index]);train_probability=model.predict_proba(values[train_index])
    metrics=compute_metrics(labels[eval_index],probability);train_metrics=compute_metrics(labels[train_index],train_probability)
    prediction_table(evaluation,labels[eval_index],probability,study.classes).to_csv(folder/'predictions.csv',index=False)
    prediction_table(training,labels[train_index],train_probability,study.classes).to_csv(folder/'training_predictions.csv',index=False)
    joblib.dump(model,folder/'pipeline.joblib',compress=3)
    result={'arm':arm,'fold':fold,'C':float(C),'phase':phase,'training_patients':len(training),'evaluation_patients':len(evaluation),
        'evaluation_f1':metrics['f1'],'training_f1':train_metrics['f1'],'f1_gap':train_metrics['f1']-metrics['f1'],
        'metrics':metrics,'training_metrics':train_metrics,'converged':True,
        'solver_iterations':model.named_steps['classifier'].n_iter_.tolist(),
        'deterministic_convex_head_single_fit_per_fold':True,'test_signals_read':False}
    write_json(folder/'result.json',result)
    complete(folder,identity,['identity.json','pipeline.joblib','predictions.csv','training_predictions.csv','result.json'])
    print(f'  training_F1={result["training_f1"]:.4f} evaluation_F1={result["evaluation_f1"]:.4f}',flush=True)
    return result


def run(study,folds,arms):
    patients,matrices,sources=load_bundle(study,arms)
    for arm in arms:
        for fold in folds:
            rows=[case(study,arm,fold,C,'inner',patients,matrices[arm],sources) for C in study.protocol['inverse_regularization_C']]
            selected=select(rows);folder=study.run/'choices'/arm;folder.mkdir(parents=True,exist_ok=True);choice=folder/f'fold_{fold}.json'
            value={'experiment_hash':study.signature,'arm':arm,'fold':fold,'C':selected['C'],'inner_f1':selected['evaluation_f1'],
                'selection':'inner_macro_f1_then_smaller_C','inner_completed_sha256':{
                    f'S{i:02}':sha256_file(study.run/'cases'/arm/f'fold_{fold}'/f'inner_S{i:02}'/'completed.json') for i in range(1,10)}}
            if choice.exists() and read_json(choice)!=value:raise RuntimeError('Frozen inner choice changed')
            write_json(choice,value)
            case(study,arm,fold,selected['C'],'outer',patients,matrices[arm],sources,choice)

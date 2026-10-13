"""Evaluasi patient-level, tanpa seleksi berdasarkan hold-out."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def aggregate_patients(labels, probabilities, groups):
    ids, truth, probs = [], [], []
    labels, groups = np.asarray(labels), np.asarray(groups)
    for hid in np.unique(groups):
        mask = groups == hid
        values = np.unique(labels[mask])
        if len(values) != 1:
            raise ValueError(f"Label segmen pasien {hid} tidak konsisten.")
        ids.append(str(hid))
        truth.append(int(values[0]))
        probs.append(probabilities[mask].mean(axis=0))
    return ids, np.asarray(truth), np.asarray(probs)


def compute_metrics(truth, probabilities):
    n_classes = probabilities.shape[1]
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(truth, pred, labels=np.arange(n_classes))
    specificity = []
    for i in range(n_classes):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(float(tn / (tn + fp)) if tn + fp else 0.0)
    auc = float(roc_auc_score(truth, probabilities, multi_class="ovr", average="macro", labels=np.arange(n_classes))) if len(np.unique(truth)) == n_classes else None
    return {"accuracy": float(accuracy_score(truth, pred)), "precision": float(precision_score(truth, pred, average="macro", labels=np.arange(n_classes), zero_division=0)), "sensitivity": float(recall_score(truth, pred, average="macro", labels=np.arange(n_classes), zero_division=0)), "specificity": float(np.mean(specificity)), "f1": float(f1_score(truth, pred, average="macro", labels=np.arange(n_classes), zero_division=0)), "auc": auc}


def predict_patient(model, loader, groups, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            probabilities.append(torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
            labels.extend(targets.numpy())
    return aggregate_patients(labels, np.concatenate(probabilities), groups)


def prediction_table(ids, truth, probabilities, class_names):
    pred = probabilities.argmax(axis=1)
    rows = []
    for i, hid in enumerate(ids):
        row = {"hospital_id": hid, "true_label": int(truth[i]), "true_class": class_names[truth[i]] if truth[i] >= 0 else None, "predicted_class": class_names[pred[i]], "correct": bool(pred[i] == truth[i]) if truth[i] >= 0 else None}
        row.update({f"probability_{cls}": float(probabilities[i, k]) for k, cls in enumerate(class_names)})
        rows.append(row)
    return pd.DataFrame(rows)


def report(truth, probabilities, class_names):
    return {"metrics": compute_metrics(truth, probabilities), "confusion_matrix": confusion_matrix(truth, probabilities.argmax(1), labels=np.arange(len(class_names))).tolist(), "classification_report": classification_report(truth, probabilities.argmax(1), labels=np.arange(len(class_names)), target_names=class_names, zero_division=0, output_dict=True)}

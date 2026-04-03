"""
metrics.py – Tính toán metrics, vẽ biểu đồ, lưu ảnh lỗi phân loại.
"""

import os
import shutil
from itertools import cycle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

import config


# ─── Sensitivity / Specificity ──────────────────────────────────────────────

def compute_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLS)))
    out = {}
    for i, cls in enumerate(config.CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[cls] = {
            'sensitivity': tp / (tp + fn) if tp + fn > 0 else 0.0,
            'specificity': tn / (tn + fp) if tn + fp > 0 else 0.0,
        }
    return out


# ─── All metrics ────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, y_probs):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred) * 100

    f1p = f1_score(y_true, y_pred, average=None,
                   labels=list(range(config.NUM_CLS)), zero_division=0)
    m['f1_macro']     = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['f1_per_class'] = {config.CLASSES[i]: f1p[i] for i in range(config.NUM_CLS)}

    try:
        m['auc_macro'] = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average='macro')
        yb = label_binarize(y_true, classes=list(range(config.NUM_CLS)))
        m['auc_per_class'] = {
            config.CLASSES[i]: roc_auc_score(yb[:, i], y_probs[:, i])
            for i in range(config.NUM_CLS)
        }
    except Exception:
        m['auc_macro']     = float('nan')
        m['auc_per_class'] = {c: float('nan') for c in config.CLASSES}

    m['sens_spec'] = compute_sensitivity_specificity(y_true, y_pred)
    m['report']    = classification_report(
        y_true, y_pred, target_names=config.CLASSES, zero_division=0)
    return m


# ─── Print metrics ──────────────────────────────────────────────────────────

def print_metrics(m, split='Test'):
    sep = '─' * 62
    print(f'\n{sep}\n  {split}\n{sep}')
    print(f'  Acc={m["accuracy"]:.2f}%  '
          f'F1={m["f1_macro"]:.4f}  '
          f'AUC={m["auc_macro"]:.4f}')
    print(f'\n  {"Class":<40} {"F1":>6}  {"AUC":>6}  {"Sens":>6}  {"Spec":>6}')
    print(f'  {"─" * 58}')
    for cls in config.CLASSES:
        print(f'  {cls:<40}'
              f' {m["f1_per_class"][cls]:>6.4f}'
              f'  {m["auc_per_class"][cls]:>6.4f}'
              f'  {m["sens_spec"][cls]["sensitivity"]:>6.4f}'
              f'  {m["sens_spec"][cls]["specificity"]:>6.4f}')
    print(f'\n{m["report"]}\n{sep}')


# ─── Plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, run_name, plots_dir):
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=ax,
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 13, 'weight': 'bold'})
    for i in range(len(config.CLASSES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#2ecc71', lw=2.5))
    ax.set_title(f'Confusion Matrix — {run_name}', fontsize=13, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    p = os.path.join(plots_dir, f'{run_name}_confusion_matrix.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'    Confusion matrix → {p}')
    return p


def plot_roc_curves(y_true, y_probs, run_name, plots_dir):
    yb     = label_binarize(y_true, classes=list(range(config.NUM_CLS)))
    colors = cycle(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'])
    fig, ax = plt.subplots(figsize=(9, 7))
    mfpr = np.linspace(0, 1, 200)
    tprs = []
    for i, (cls, col) in enumerate(zip(config.CLASSES, colors)):
        fpr, tpr, _ = roc_curve(yb[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=col, lw=1.8,
                label=f'{cls} (AUC={auc(fpr, tpr):.3f})')
        tprs.append(np.interp(mfpr, fpr, tpr))
    mt = np.mean(tprs, axis=0)
    ax.plot(mfpr, mt, 'k--', lw=2.2,
            label=f'Macro (AUC={auc(mfpr, mt):.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel='FPR', ylabel='TPR',
           title=f'ROC Curves — {run_name}')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    p = os.path.join(plots_dir, f'{run_name}_roc_curves.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'    ROC curves → {p}')
    return p


def plot_sensitivity_specificity(m, run_name, plots_dir):
    sens = [m['sens_spec'][c]['sensitivity'] for c in config.CLASSES]
    spec = [m['sens_spec'][c]['specificity'] for c in config.CLASSES]
    x, w = np.arange(len(config.CLASSES)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, sens, w, label='Sensitivity',
           color='#4292c6', edgecolor='white')
    ax.bar(x + w/2, spec, w, label='Specificity',
           color='#41ab5d', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASSES, rotation=20, ha='right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.set_title(f'Sensitivity & Specificity — {run_name}', fontsize=13)
    for rect, val in zip(ax.patches, sens + spec):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p = os.path.join(plots_dir, f'{run_name}_sens_spec.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'   Sens/Spec → {p}')
    return p


# ─── Save misclassified images ───────────────────────────────────────────────

def save_errors(y_true, y_pred, paths, seed, error_dir):
    d = os.path.join(error_dir, f'Seed{seed}')
    os.makedirs(d, exist_ok=True)
    n = 0
    for t, p, path in zip(y_true, y_pred, paths):
        if t != p:
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            ext = os.path.splitext(path)[1]
            
            true_label = config.IDX_TO_CLS[t].replace(' ', '_')
            pred_label = config.IDX_TO_CLS[p].replace(' ', '_')
            
            fname = f"{base_name}_true_{true_label}_predict_{pred_label}{ext}"
            
            shutil.copy2(path, os.path.join(d, fname))
            n += 1
    print(f'    Misclassified: {n}  (saved → {d})')
    return n

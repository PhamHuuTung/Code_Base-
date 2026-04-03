"""
wandb_utils.py – Các helper function cho Weights & Biases logging.
"""

import os

import wandb
from sklearn.metrics import confusion_matrix

import config


def wandb_setup():
    """Đăng nhập W&B và in trạng thái."""
    os.environ['WANDB_API_KEY'] = config.WANDB_API_KEY
    os.environ['WANDB_MODE']    = config.WANDB_MODE
    if config.WANDB_MODE != 'disabled':
        wandb.login(relogin=True, force=True)
        print(f' W&B  mode={config.WANDB_MODE}  project={config.PROJECT_NAME}')
    else:
        print(' W&B disabled')


def make_wandb_config(seed: int) -> dict:
    return {
        'model_name':    config.MODEL_NAME,
        'model_pretrain': config.MODEL_PRETRAIN,
        'data_dir':      config.DATA_DIR,
        'classes':       config.CLASSES,
        'num_classes':   config.NUM_CLS,
        'aug_mode':      config.AUG_MODE,
        **config.HPARAMS,
        'seed': seed,
    }


def wandb_log_epoch(epoch, train_loss, val_loss, val_acc, val_f1):
    if config.WANDB_MODE == 'disabled':
        return
    wandb.log({'epoch':      epoch,
               'train_loss': train_loss,
               'val_loss':   val_loss,
               'val_acc':    val_acc,
               'val_f1':     val_f1})


def wandb_log_test(metrics: dict, test_loss: float):
    if config.WANDB_MODE == 'disabled':
        return
    payload = {
        'test_loss': test_loss,
        'test_acc':  metrics['accuracy'],
        'test_f1':   metrics['f1_macro'],
        'test_auc':  metrics['auc_macro'],
    }
    for cls in config.CLASSES:
        s = cls.replace(' ', '_')
        payload[f'test_f1_{s}']   = metrics['f1_per_class'][cls]
        payload[f'test_auc_{s}']  = metrics['auc_per_class'][cls]
        payload[f'test_sens_{s}'] = metrics['sens_spec'][cls]['sensitivity']
        payload[f'test_spec_{s}'] = metrics['sens_spec'][cls]['specificity']
    wandb.log(payload)


def wandb_log_images(cm_path: str, roc_path: str, ss_path: str):
    if config.WANDB_MODE == 'disabled':
        return
    wandb.log({
        'confusion_matrix_img':    wandb.Image(cm_path),
        'roc_curves':              wandb.Image(roc_path),
        'sensitivity_specificity': wandb.Image(ss_path),
    })


def wandb_log_confusion_matrix(y_true, y_pred):
    if config.WANDB_MODE == 'disabled':
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLS)))
    table = wandb.Table(columns=['True \\ Pred'] + config.CLASSES)
    for i, row_cls in enumerate(config.CLASSES):
        table.add_data(row_cls, *[int(cm[i, j]) for j in range(config.NUM_CLS)])
    wandb.log({'confusion_matrix_table': table})

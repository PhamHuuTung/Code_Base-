"""
train.py – File chính để chạy training pipeline.

Cách chạy:
  # Dùng cấu hình trong config.py
  python train.py
  r"D:\TUNGG\ml_project\.venv\Scripts\Activate.ps1"
  # Ghi đè tham số từ terminal
  python train.py --model densenet121 --epochs 30 --lr 0.0001
  python train.py --model efficientnet_b0 --batch_size 32 --seeds 42 1234
  python train.py --wandb_mode disabled
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler

import config
import data
import engine
import metrics
import model as model_builder
import wandb_utils

warnings.filterwarnings('ignore')

# ─── Device ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── CLI arguments ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Ear Classification Training Pipeline')
    p.add_argument('--model',       type=str,   default=None,
                   help='timm model name, e.g. resnet50, densenet121')
    p.add_argument('--epochs',      type=int,   default=None)
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--seeds',       type=int,   nargs='+', default=None,
                   help='Một hoặc nhiều seeds, e.g. --seeds 42 1234')
    p.add_argument('--aug_mode',    type=str,   default=None,
                   choices=['balanced', 'imbalanced', 'none'])
    p.add_argument('--data_dir',    type=str,   default=None,
                   help='Đường dẫn thư mục data')
    p.add_argument('--wandb_mode',  type=str,   default=None,
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--no_pretrain', action='store_true',
                   help='Không dùng pretrained weights')
    return p.parse_args()


def apply_args(args):
    """Ghi đè config bằng CLI arguments (nếu được cung cấp)."""
    if args.model:       config.MODEL_NAME      = args.model
    if args.epochs:      config.HPARAMS['epochs']     = args.epochs
    if args.batch_size:  config.HPARAMS['batch_size']  = args.batch_size
    if args.lr:          config.HPARAMS['lr']           = args.lr
    if args.seeds:       config.HPARAMS['seeds']        = args.seeds
    if args.aug_mode:    config.AUG_MODE          = args.aug_mode
    if args.data_dir:    config.DATA_DIR           = args.data_dir
    if args.wandb_mode:  config.WANDB_MODE         = args.wandb_mode
    if args.no_pretrain: config.MODEL_PRETRAIN     = False


# ─── Seed ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Setup dirs ─────────────────────────────────────────────────────────────

def setup_dirs():
    for d in config.DIRS.values():
        os.makedirs(d, exist_ok=True)
    print(' Output directories:')
    for k, v in config.DIRS.items():
        print(f'   {k:<8}: {v}')


# ─── Single seed training ────────────────────────────────────────────────────

def train_single_seed(seed: int, all_paths, all_labels, train_tf, val_tf):
    model_tag  = config.MODEL_NAME.replace('/', '_')
    run_name   = f'{model_tag}_Seed{seed}'
    model_path = os.path.join(config.DIRS['models'], f'{run_name}.pth')

    if os.path.exists(model_path):
        print(f'⏭  Skipping Seed {seed} — checkpoint exists: {model_path}')
        return

    print(f"\n{'='*60}")
    print(f'  Seed {seed}  |  {run_name}')
    print(f"{'='*60}")
    set_seed(seed)

    # DataLoaders
    loader_train, loader_val, loader_test = data.build_dataloaders(
        all_paths, all_labels, train_tf, val_tf, seed)

    # Model, optimizer, loss
    mdl       = model_builder.create_model()
    optimizer = torch.optim.AdamW(
        mdl.parameters(),
        lr=config.HPARAMS['lr'],
        weight_decay=config.HPARAMS['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)
    scaler    = GradScaler(enabled=config.HPARAMS['use_amp'])

    # W&B init
    if config.WANDB_MODE != 'disabled':
        wandb.init(project=config.PROJECT_NAME, name=run_name,
                   config=wandb_utils.make_wandb_config(seed), reinit=True)

    best_loss, patience_ctr = float('inf'), 0

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(config.HPARAMS['epochs']):

        train_loss = engine.train_one_epoch(
            mdl, loader_train, criterion, optimizer, scaler,
            epoch, config.HPARAMS['use_amp'], config.HPARAMS['print_freq'])

        yt_v, yp_v, _, _, val_loss, val_acc = engine.predict(
            mdl, loader_val, criterion, config.HPARAMS['print_freq'])
        val_f1 = f1_score(yt_v, yp_v, average='macro', zero_division=0)

        print(f'  [Epoch {epoch:03d}]  '
              f'train={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_acc={val_acc:.2f}%  '
              f'val_f1={val_f1:.4f}')

        wandb_utils.wandb_log_epoch(epoch, train_loss, val_loss, val_acc, val_f1)

        # Early stopping (theo train_loss như code gốc)
        if train_loss < best_loss:
            best_loss     = train_loss
            patience_ctr  = 0
            engine.save_checkpoint({
                'epoch':                epoch + 1,
                'seed':                 seed,
                'model_state_dict':     mdl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_loss,
                'val_acc':              val_acc,
                'val_f1':               val_f1,
            }, model_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.HPARAMS['patience']:
                print(f'  Early stopping tại epoch {epoch}')
                break

    # ── Test set ────────────────────────────────────────────────────────────
    print('\n  Evaluating on test set...')
    mdl, _, _, _ = engine.load_checkpoint(model_path, mdl)
    yt, yp, ypr, pths, test_loss, _ = engine.predict(
        mdl, loader_test, criterion, config.HPARAMS['print_freq'])

    m = metrics.compute_all_metrics(yt, yp, ypr)
    metrics.print_metrics(m, split=f'Test – Seed {seed}')

    # ── Lưu report text ─────────────────────────────────────────────────────
    rpt_path = os.path.join(config.DIRS['reports'], f'{run_name}_report.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(f'Run: {run_name}  Seed: {seed}\n{"─"*60}\n')
        f.write(f'Acc={m["accuracy"]:.2f}%  '
                f'F1={m["f1_macro"]:.4f}  '
                f'AUC={m["auc_macro"]:.4f}\n\n')
        f.write(f'{"Class":<42} {"F1":>6}  {"AUC":>6}  '
                f'{"Sens":>6}  {"Spec":>6}\n{"─"*60}\n')
        for cls in config.CLASSES:
            f.write(f'{cls:<42}'
                    f' {m["f1_per_class"][cls]:>6.4f}'
                    f'  {m["auc_per_class"][cls]:>6.4f}'
                    f'  {m["sens_spec"][cls]["sensitivity"]:>6.4f}'
                    f'  {m["sens_spec"][cls]["specificity"]:>6.4f}\n')
        f.write(f'\n{m["report"]}')
    print(f'    Report → {rpt_path}')

    # ── Plots ───────────────────────────────────────────────────────────────
    cm_p  = metrics.plot_confusion_matrix(yt, yp,  run_name, config.DIRS['plots'])
    roc_p = metrics.plot_roc_curves(yt, ypr,        run_name, config.DIRS['plots'])
    ss_p  = metrics.plot_sensitivity_specificity(m,  run_name, config.DIRS['plots'])

    # ── W&B logging ─────────────────────────────────────────────────────────
    wandb_utils.wandb_log_test(m, test_loss)
    wandb_utils.wandb_log_images(cm_p, roc_p, ss_p)
    wandb_utils.wandb_log_confusion_matrix(yt, yp)

    # ── Lưu ảnh phân loại sai ───────────────────────────────────────────────
    metrics.save_errors(yt, yp, pths, seed, config.DIRS['errors'])

    if config.WANDB_MODE != 'disabled':
        wandb.finish()

    print(f'  Seed {seed} hoàn tất.\n')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    apply_args(args)

    # In cấu hình
    print('='*60)
    print(f'  Project : {config.PROJECT_NAME}')
    print(f'  Data    : {config.DATA_DIR}')
    print(f'  Model   : {config.MODEL_NAME}  pretrained={config.MODEL_PRETRAIN}')
    print(f'  Aug     : {config.AUG_MODE}')
    print(f'  Seeds   : {config.HPARAMS["seeds"]}')
    print(f'  Epochs  : {config.HPARAMS["epochs"]}')
    print(f'  W&B     : {config.WANDB_MODE}')
    print(f'  Device  : {device}')
    if device.type == 'cuda':
        import torch.cuda
        print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    print('='*60)

    setup_dirs()
    wandb_utils.wandb_setup()

    # Load data
    print('\n Loading dataset...')
    all_paths, all_labels = data.load_data(config.DATA_DIR)

    # Transforms
    train_tf, val_tf = data.get_transforms()

    # Train mỗi seed
    for seed in config.HPARAMS['seeds']:
        train_single_seed(seed, all_paths, all_labels, train_tf, val_tf)

    print('\n  All seeds finished.')


if __name__ == '__main__':
    main()

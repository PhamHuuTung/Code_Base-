"""
engine.py – Training loop, predict, checkpoint save/load.
"""

import os
import time

import numpy as np
import torch
from enum import Enum
from torch.cuda.amp import GradScaler, autocast

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Utility meters ─────────────────────────────────────────────────────────

class Summary(Enum):
    NONE    = 0
    AVERAGE = 1
    SUM     = 2
    COUNT   = 3


class AverageMeter:
    def __init__(self, name, fmt=':f', st=Summary.AVERAGE):
        self.name, self.fmt, self.st = name, fmt, st
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def __str__(self):
        return ('{name} {val' + self.fmt + '} ({avg' + self.fmt + '})').format(
            **self.__dict__)

    def summary(self):
        if self.st is Summary.NONE:    return ''
        if self.st is Summary.AVERAGE: return f'{self.name} {self.avg:.4f}'
        if self.st is Summary.SUM:     return f'{self.name} {self.sum:.4f}'
        return ''


class ProgressMeter:
    def __init__(self, nb, meters, prefix=''):
        nd = len(str(nb))
        self.fmt    = '[{:' + str(nd) + 'd}/' + str(nb) + ']'
        self.meters = meters
        self.prefix = prefix

    def display(self, b):
        print('\t'.join(
            [self.prefix + self.fmt.format(b)] + [str(m) for m in self.meters]))

    def display_summary(self):
        print(' '.join(
            [' *'] + [m.summary() for m in self.meters if m.summary()]))


# ─── Forward pass ────────────────────────────────────────────────────────────

def forward(model, imgs: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with autocast(enabled=use_amp):
        logits = model(imgs)
    return logits


# ─── Train one epoch ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    epoch: int, use_amp: bool, pf: int) -> float:
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    top1   = AverageMeter('Acc',  ':6.2f', Summary.NONE)
    bt     = AverageMeter('Time', ':6.3f', Summary.NONE)
    prog   = ProgressMeter(len(loader), [bt, losses, top1],
                           prefix=f'Epoch [{epoch}]')
    model.train()
    end = time.time()

    for i, (imgs, labels, _) in enumerate(loader):
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = forward(model, imgs, use_amp)
        loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), imgs.size(0))
        top1.update(
            (logits.argmax(1) == labels).float().mean().item() * 100,
            imgs.size(0))
        bt.update(time.time() - end); end = time.time()
        if i % pf == 0:
            prog.display(i + 1)

    prog.display_summary()
    return losses.avg


# ─── Predict (val / test / inference) ───────────────────────────────────────

def predict(model, loader, criterion, pf: int):
    """
    Hỗ trợ 2 loại loader:
      • CustomDataset   (imgs, labels, path)  → tính loss / acc
      • InferenceDataset (imgs, path)         → chỉ trả probs / preds
    """
    losses = AverageMeter('Loss', ':.4e',  Summary.AVERAGE)
    top1   = AverageMeter('Acc',  ':6.2f', Summary.AVERAGE)
    bt     = AverageMeter('Time', ':6.3f', Summary.NONE)
    prog   = ProgressMeter(len(loader), [bt, losses, top1], prefix='Predict: ')

    model.eval()
    y_true, y_pred, y_probs, paths = [], [], [], []
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(batch) == 3:                        # có nhãn
                imgs, labels, p = batch
                imgs   = imgs.to(device,   non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                has_labels = True
            else:                                      # không nhãn
                imgs, p = batch
                imgs = imgs.to(device, non_blocking=True)
                labels     = None
                has_labels = False

            logits = forward(model, imgs, use_amp=False)
            y_probs.extend(torch.softmax(logits, 1).cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            paths.extend(p)

            if has_labels:
                y_true.extend(labels.cpu().numpy())
                if criterion is not None:
                    loss = criterion(logits, labels)
                    losses.update(loss.item(), imgs.size(0))
                top1.update(
                    (logits.argmax(1) == labels).float().mean().item() * 100,
                    imgs.size(0))

            bt.update(time.time() - end); end = time.time()
            if i % pf == 0:
                prog.display(i + 1)

    prog.display_summary()
    return (
        np.array(y_true) if y_true else None,
        np.array(y_pred),
        np.array(y_probs),
        paths,
        losses.avg,
        top1.avg,
    )


# ─── Checkpoint ─────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f'    Saved → {path}')


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    ep = ckpt.get('epoch', 0)
    bl = ckpt.get('best_val_loss', float('inf'))
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f'   => Loaded  epoch={ep}  best_loss={bl:.4f}')
    return model, optimizer, ep, bl

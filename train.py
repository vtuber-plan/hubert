import os
import json
import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from hubert.data.dataset import AcousticUnitsDataset
from hubert.model.hubert import Hubert


def train():
    model = Hubert()
    
    train_dataset = AcousticUnitsDataset("datasets/genshin/train", sample_rate=48000, label_rate=50)
    valid_dataset = AcousticUnitsDataset("datasets/genshin/valid", sample_rate=48000, label_rate=50)

    collate_fn = AcousticUnitsCollate()
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[3],
        # strategy="ddp",
        # amp_backend="native",
        # precision=16,
        # logger=logger,
        # max_steps=100,
        max_epochs=20000,
        default_root_dir="./logs",
        limit_val_batches=1
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    train()
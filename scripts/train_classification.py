import argparse
import os
import pydoc
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from train_utils import get_classification_transforms as get_transforms

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import BirdClassificationDataset, CustomSubset
from src.utils import fix_seed, worker_init_fn


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--config_path", type=str, default="./config/base_classification.yaml"
    )
    parser.add_argument("--devices", type=int, nargs="+")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

        if args.devices is not None:
            hparams["trainer"]["devices"] = (
                args.devices if len(args.devices) > 0 else [args.devices]
            )
        logger.info(f'devices: {hparams["trainer"]["devices"]}')

    os.makedirs(hparams["output_root_dir"], exist_ok=True)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    val_dataset = BirdClassificationDataset(
        hparams["dataset"]["data_root"],
        split="val",
    )

    train_dataset = BirdClassificationDataset(
        hparams["dataset"]["data_root"],
        split="train",
    )

    train_labels = train_dataset.df["label"]
    train_class_counts = np.bincount(train_labels)
    train_class_weights = 1.0 / train_class_counts
    train_sample_weights = train_class_weights[train_labels]
    train_sample_weights = torch.DoubleTensor(train_sample_weights)
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True,
    )

    transforms_dict = get_transforms(hparams)

    train_dataset = CustomSubset(
        train_dataset,
        transforms_dict["train"],
    )

    val_dataset = CustomSubset(
        val_dataset,
        transforms_dict["val"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["train_parameters"]["batch_size"],
        sampler=train_sampler,
        drop_last=True,
        num_workers=hparams["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["val_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
    )

    trainer_params = hparams["trainer"]
    trainer = Trainer(
        default_root_dir=hparams["output_root_dir"],
        max_epochs=trainer_params["max_epochs"],
        log_every_n_steps=trainer_params["log_every_n_steps"],
        devices=trainer_params["devices"],
        accelerator=trainer_params["accelerator"],
        gradient_clip_val=trainer_params["gradient_clip_val"],
        accumulate_grad_batches=trainer_params["accumulate_grad_batches"],
        deterministic="warn",
        num_sanity_val_steps=trainer_params["num_sanity_val_steps"],
        logger=TensorBoardLogger(
            save_dir=hparams["output_root_dir"],
            version=f"{hparams['experiment_name']}"
            f"{hparams['train_parameters']['batch_size']*trainer_params['accumulate_grad_batches']}_"
            f"{hparams['optimizer']['lr']}",
            name=f"{hparams['experiment_name']}",
        ),
        callbacks=[
            ModelCheckpoint(
                monitor=trainer_params["model_check_point"]["monitor"],
                mode=trainer_params["model_check_point"]["mode"],
                save_top_k=1,
                verbose=True,
                filename="{epoch:03d}_{step:05d}_{val_acc:.4f}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        precision=trainer_params["precision"],
    )

    if trainer_params["resume_from_checkpoint"] is not None and os.path.isfile(
        trainer_params["resume_from_checkpoint"]
    ):
        trainer.fit(
            pydoc.locate(hparams["model"]["pl_class"])(hparams),
            train_loader,
            val_loader,
            ckpt_path=trainer_params["resume_from_checkpoint"],
        )
    else:
        trainer.fit(
            pydoc.locate(hparams["model"]["pl_class"])(hparams),
            train_loader,
            val_loader,
        )


if __name__ == "__main__":
    main()

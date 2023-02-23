import argparse
import os
from omegaconf import OmegaConf
from pathlib import Path
from time import time

import torch

from lightning import Trainer, seed_everything
from lightning.pytorch import callbacks, loggers, strategies

import wandb.util

from helpers import min_max

from datamodule import CityscapesDataModule
from lightning_model import LitSemanticSegmentationModel


def get_rank() -> int:
    """
    Determines the rank of the process in the case where a multi-GPUs job would be started.
    :return:
    """
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def dump(data, path: Path):
    """
    Saves an OmegaConf dict to disk

    :param data:
    :param path:
    :return:
    """
    with path.open("w") as file:
        OmegaConf.save(data, file)


def load(path: Path):
    """
    Loads an OmegaConf dict from disk

    :param path:
    :return:
    """
    with path.open("r") as file:
        data = OmegaConf.load(file)
    return data


def hparams_from_file(path: Path):
    """
    Reads the hyperparameters from a `.yaml` file.

    :param path:
    :return:
    """
    return load(path)


def hparams_from_checkpoint(path: Path) -> dict:
    """
    Reads the hyperparameters from within a checkpoint.

    :param path:
    :return:
    """
    checkpoint = torch.load(path)
    hparams = checkpoint["hyper_parameters"]
    return hparams


def config_from_hparams(hparams):
    """
    By fetching information about the CPUs, GPUs, it determines the current machine configuration, e.g.
    the batch size per GPU, whether gradient accumulation is needed, the amount of workers for dataloading per GPU.

    :param hparams:
    :return:
    """
    config = OmegaConf.create()
    config.num_cpus = os.cpu_count()
    config.num_gpus = torch.cuda.device_count()
    config.num_workers = min_max((config.num_cpus - 1) // config.num_gpus, m=1, M=8)

    assert hparams.optim.batch_size % config.num_gpus == 0

    config.batch_size_per_gpu = min_max(
        min(
            hparams.trainer.max_batch_size_per_gpu,
            hparams.optim.batch_size // config.num_gpus,
        ),
        m=1,
    )
    config.accumulate_grad_batches = min_max(
        hparams.optim.batch_size // (config.num_gpus * config.batch_size_per_gpu),
        m=1,
    )
    assert (
        config.num_gpus * config.batch_size_per_gpu * config.accumulate_grad_batches
        == hparams.optim.batch_size
    )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Path to an existing checkpoint."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Path to the dataset root directory."
    )
    parser.add_argument(
        "--offline",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enables the logging online on W&B.",
    )
    args = parser.parse_args()

    experiment = args.experiment
    checkpoint = args.checkpoint
    root = args.root
    offline = args.offline

    if get_rank() == 0:
        # If the current process is the rank zero, then look at the machine specs.
        hparams = hparams_from_file(path=experiment / "hparams.yaml")
        config = config_from_hparams(hparams=hparams)

        if hparams.trainer.id is None:
            # If the experiment id is not specified we generate one.
            hparams.trainer.id = wandb.util.generate_id()

        if (experiment / "checkpoints" / "last.ckpt").exists():
            # If a checkpoint folder exists, then we will automatically restart from it.
            checkpoint = experiment / "checkpoints" / "last.ckpt"
            hparams.trainer.id = hparams_from_checkpoint(checkpoint).trainer.id
            config.resume = True
        else:
            config.resume = False

        print(OmegaConf.to_yaml(hparams))
        print(OmegaConf.to_yaml(config))

        if config.num_gpus > 1:
            dump(hparams, Path.cwd() / "hparams.temp.yaml")
            dump(config, Path.cwd() / "config.temp.yaml")
    else:
        # Otherwise we load them from disk.
        hparams = load(Path.cwd() / "hparams.temp.yaml")
        config = load(Path.cwd() / "config.temp.yaml")

    torch.set_num_threads(config.num_cpus)

    seed_everything(hparams.trainer.seed, workers=True)

    if checkpoint is None or config.resume:
        print("DEFAULT INITIALISATION")
        model = LitSemanticSegmentationModel(hparams=hparams)
    else:
        print("DEFAULT LOAD FROM CHECKPOINT")
        model = LitSemanticSegmentationModel.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location="cpu",
            hparams=hparams,
            strict=False,
        )

    datamodule = CityscapesDataModule(
        hparams=hparams,
        root=root,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.num_workers,
    )

    # We create a logger to monitor the training.
    save_dir = Path(f"/tmp/TEMP_{time()}")
    save_dir.mkdir(parents=True)
    logger = loggers.WandbLogger(
        name=f"{experiment.name}",
        save_dir=f"{save_dir}",
        id=hparams.trainer.id,
        project="aagi_assignment",
        log_model=False,
        save_code=False,
        offline=offline,
    )

    # We create various callbacks which will plug in automatically during training.
    model_checkpoint = callbacks.ModelCheckpoint(
        dirpath=experiment / "checkpoints",
        monitor="val_metric",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
    )
    learning_rate_monitor = callbacks.LearningRateMonitor()
    tqdm_progress_bar = callbacks.TQDMProgressBar()
    model_summary = callbacks.ModelSummary()

    # We choose the training strategy to adopt in case of distributed training across multiple GPUs.
    if config.num_gpus > 1:
        strategy = strategies.DDPStrategy()
    else:
        strategy = None

    # We setup the Trainer.
    trainer = Trainer(
        default_root_dir=f"{experiment}",
        accelerator="gpu",
        devices=config.num_gpus,
        auto_select_gpus=True,
        strategy=strategy,
        detect_anomaly=True,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        precision=hparams.trainer.precision,
        logger=logger,
        callbacks=[
            model_checkpoint,
            learning_rate_monitor,
            tqdm_progress_bar,
            model_summary,
        ],
        log_every_n_steps=16 * config.accumulate_grad_batches,
        max_epochs=hparams.optim.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        sync_batchnorm=True if config.num_gpus > 1 else False,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    # We start training the model.
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint if config.resume else None,
    )

    (Path.cwd() / "hparams.temp.yaml").unlink(missing_ok=True)
    (Path.cwd() / "config.temp.yaml").unlink(missing_ok=True)

    print(model_checkpoint.best_model_path)


if __name__ == "__main__":
    main()

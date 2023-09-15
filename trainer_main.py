from tkinter import Image
import os
import pytorch_lightning as pl
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, \
    ModelSummary, LearningRateMonitor
from argparse import ArgumentParser

from tailor import Tailor
from imagenet_datamodule import ImagenetDataModule
from torchinfo import summary 
import torch

def print_model(model, input_data):
    """
    View model using torch summary
    
    How to get input size shape from dataloader:
    print_model(model, next(iter(train_loader))[0].shape)

    Though this time we might pass actual data (next(iter(train_loader))[0]) instead...
    """

    summary(
        model, 
        input_size=input_data, 
        depth=9,
        col_names=['input_size', 'output_size', 'num_params'], 
        row_settings=['var_names']
    )

def main(args):
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(args.random_seed, workers=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Prepare data
    if args.dataset == 'imagenet': 
        data_module = ImagenetDataModule.from_argparse_args(args)
    elif args.dataset == 'cifar10': 
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        data_module = CIFAR10DataModule(
            data_dir='.',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
    else: 
        raise RuntimeError("Unsupported dataset")
    

    # Initialize Tailor training settings
    tailor = Tailor(**vars(args))

    data_module.prepare_data()
    data_module.setup()
    print_model(tailor.teacher, next(iter(data_module.train_dataloader()))[0].shape)

    # Set up Trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        args.save_dir, 
        name=args.experiment_name
    )

    # Set up model checkpointing
    # Save top-3 checkpoints based on Val/Loss
    top3_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        monitor='Val/Loss',
        mode='min',
        dirpath=os.path.join(args.save_dir, args.experiment_name),
        filename=f'{args.experiment_name}'
            '_epoch={epoch:02d}_loss={Val/Loss:.2f}'
            '_top1={Val/Top1:.2f}',
        auto_insert_metric_name=False
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    find_unused_parameters = args.find_unused_parameters 
    # -----------------------^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # | TODO: Set to True if training with modifiers and/or short skips
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.num_gpu,
        logger=tb_logger,
        max_epochs=args.num_epochs,
        strategy=pl.strategies.DDPStrategy(
            find_unused_parameters=find_unused_parameters
        ),
        callbacks=[
            lr_monitor, 
            top3_checkpoint_callback, 
            ModelSummary(max_depth=-1),
            # EarlyStopping(monitor="Val/Loss", mode="min")
        ],
        auto_select_gpus=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=args.fast_dev_run,
    )
    print(f"Checkpoint resume path = {args.checkpoint_resume_path}")
    trainer.fit(
        model=tailor, 
        datamodule=data_module, 
        ckpt_path=args.checkpoint_resume_path
    )

    # Run best model on validation set and write results to file
    val_results = trainer.validate(
        model=tailor, 
        datamodule=data_module, 
        ckpt_path='best'
    )
    val_results_log = os.path.join(
        args.save_dir, 
        args.experiment_name,
        args.experiment_name + '_validation.txt'
    )
    with open(val_results_log, "w") as f:
        f.write(str(val_results))
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=-1) 
    # By default, use all available GPUs --------------^^
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument(
        '--accelerator', 
        type=str, 
        choices=['cpu', 'gpu', 'auto'], 
        default='gpu'
    )
    parser.add_argument('--fast_dev_run', action='store_true', default=False)
    parser.add_argument('--find_unused_parameters', action='store_true', default=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'cifar10'], default='imagenet')
    
    # Add dataset-specific args
    parser = ImagenetDataModule.add_argparse_args(parser)
    # Add Tailor-specific args
    parser = Tailor.add_model_specific_args(parser)
    
    args = parser.parse_args()
    main(args)
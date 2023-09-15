import os
import ffcv
import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.datasets   as datasets
import torchvision.transforms as transforms


class ImagenetDataModule(pl.LightningDataModule):
    """
    - 1000 classs
    - Each image is 3 x 224 x 224

    This class applies the standard transforms to the ImageNet training and
    validation datasets.
    """
    def __init__(
        self, 
        data_dir, 
        num_workers=24, # num_workers should be <= num_cpu / num_gpu
        batch_size=256, 
        pin_memory=True, # Set True for KD
        drop_last=False, # Turn on for multi-gpu training
        ffcv=False,
    ):
        super().__init__()
        self.num_classes = 1000
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.ffcv = ffcv

        if self.ffcv:
            self.train_dir = os.path.join(self.data_dir, 'imagenet_train.ffcv') 
            self.val_dir = os.path.join(self.data_dir, 'imagenet_val.ffcv')
        else: 
            self.train_dir = os.path.join(self.data_dir, 'train') 
            self.val_dir = os.path.join(self.data_dir, 'val')
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('ImageNet')
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--num_workers', type=int, default=24)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--pin_memory', action='store_true', default=False)
        parser.add_argument('--drop_last', action='store_true', default=False)
        # NOTE: Set drop_last to True if multi-gpu training
        parser.add_argument('--ffcv', action='store_true', default=False)
        return parent_parser

    def prepare_data(self):
        """
        Download ImageNet on your own. Google how to setup. It will take a
        while. You'll have to fix the crane_bird vs crane (machinery) bug. A
        downloadable devkit might help with that. 
        """
        pass 

    def train_dataloader(self):
        """
        Return ImageNet training dataloader
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        if self.ffcv:
            # NOTE: get_rank() only works for single GPU environments.
            this_device = torch.device(f'cuda:{torch.distributed.get_rank()}')
            label_pipeline = [
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
            ]
            train_image_pipeline = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((224, 224)),
                ffcv.transforms.RandomHorizontalFlip(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(this_device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.Convert(torch.float32),
                normalize,
            ]
            order = ffcv.loader.OrderOption.QUASI_RANDOM
            return ffcv.loader.Loader(
                self.train_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=order,
                os_cache=False, 
                # ^^ Set to True if we have enough RAM to fit entire dataset
                drop_last=self.drop_last,
                pipelines={
                    'image': train_image_pipeline,
                    'label': label_pipeline,
                },
            )
        # Else use standard PyTorch dataloader
        train_dataset = datasets.ImageFolder(
            self.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                normalize,
            ])
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle during training
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=None
        )

    def val_dataloader(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        if self.ffcv:
            cropper = ffcv.fields.rgb_image.CenterCropRGBImageDecoder(
                (256, 256), 
                ratio=224/256
            )
            # NOTE: get_rank() only works for single GPU environments.
            this_device = torch.device(f'cuda:{torch.distributed.get_rank()}')
            label_pipeline = [
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
            ]
            val_image_pipeline = [
                cropper,
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(this_device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.Convert(torch.float32),
                normalize,
            ]
            return ffcv.loader.Loader(
                self.val_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=ffcv.loader.OrderOption.SEQUENTIAL,
                drop_last=self.drop_last,
                pipelines={
                    'image': val_image_pipeline,
                    'label': label_pipeline,
                },
            )
        # Else use standard PyTorch dataloader
        val_dataset = datasets.ImageFolder(
            self.val_dir, 
            transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(), 
                normalize
            ])
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Don't shuffle during validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        """
        ImageNet uses the validation dataset as the test dataset.
        """
        return self.val_dataloader()

    def predict_dataloader(self):
        # Use validation dataset for now for single predictions
        return self.val_dataloader()
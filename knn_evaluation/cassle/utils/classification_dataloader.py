
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from torch.utils.data.dataset import Subset

import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder, MNIST
from cassle.utils.datasets import DomainNetDataset
from sklearn.model_selection import train_test_split


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }
    return pipeline

class channel_mul_transform(nn.Module):
    
    def forward(self, image_tensor):
        
        final = torch.cat([image_tensor, image_tensor, image_tensor], dim = 0)
    
        return final      
        

def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """
    mnist_pipeline = {
        "T_train" :transforms.Compose(
                [transforms.RandomResizedCrop(size=28, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                channel_mul_transform(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ,
        "T_val" : transforms.Compose(
                [transforms.ToTensor(),
                channel_mul_transform(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                
    }
            
    
        
    
    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(32),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(32),  # resize shorter
                transforms.CenterCrop(32),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "mnist" : mnist_pipeline,
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "domainnet": imagenet_pipeline,
        "custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    train_domain: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[Union[str, Path]]: path where to download/locate the dataset.
        train_dir Optional[Union[str, Path]]: subpath where the training data is located.
        val_dir Optional[Union[str, Path]]: subpath where the validation data is located.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = Path(f"{dataset}/val")
    else:
        val_dir = Path(val_dir)

    assert dataset in [
        "mnist",
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
        "domainnet",
        "custom",
    ]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=True,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=False,
            download=True,
            transform=T_val,
        )
    elif dataset == "mnist":
        
        train_dataset = MNIST(
            data_dir /  train_dir,
            train=True,
            download=True,
            transform= T_train)
        
        val_dataset = MNIST(
            data_dir / val_dir,
            train=False,
            download=True,
            transform= T_val)

    elif dataset == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            data_dir / val_dir,
            split="test",
            download=True,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "custom"]:
        train_dir = data_dir / train_dir
        val_dir = data_dir / val_dir

        train_dataset = ImageFolder(train_dir, T_train)
        

        
        val_dataset = ImageFolder(val_dir, T_val)
        
    

    elif dataset == "domainnet":
        train_dataset = DomainNetDataset(
            data_root=data_dir,
            image_list_root=data_dir,
            domain_names=train_domain,
            split="train",
            transform=T_train,
        )
        val_dataset = DomainNetDataset(
            data_root=data_dir,
            image_list_root=data_dir,
            domain_names=None,
            split="test",
            transform=T_val,
            return_domain=True,
        )

    return train_dataset, val_dataset
import pandas as pd
from torchvision.io import read_image
class custom_imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform :
            self.transform = transform
        table = pd.read_csv("/home/sol-ex-inanis/Data/tiny-imagenet-200/val/val_annotations.txt", sep="\t")
        
        # print(table[["sample","class"]])
        
        unique_targets = list(set(list(table["class"])))[:100]
        
        print("len(unique_targets): ", len(unique_targets))
        
        # [ print(s) for s in table[["sample","class"]].to_numpy() ]
        
        class_to_idx = { target : idx for idx, target in enumerate(unique_targets)   }
        # print(class_to_idx)
        samples = [ str(s[0]) for s in table[["sample","class"]].to_numpy() if s[1] in unique_targets ]
        targets = [ class_to_idx[str(s[1])] for s in table[["sample","class"]].to_numpy() if s[1] in unique_targets ]
        
        
        
        # print("len(samples): ",len(samples))
        # print("len(targets): ", len(targets))
        
        
        
        
        
        self.root = root_dir
        print(self.root)

        self.data = samples
        self.targets = targets
        self.classes = unique_targets
        
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        data_path = os.path.join(self.root, self.data[idx])
        sample = read_image(data_path)
        if sample.size(0) == 1 :
            sample = torch.cat([sample for i in range(3)], dim = 0)
        target = self.targets[idx]
        # sample = sample.reshape( (sample.shape[2], sample.shape[0], sample.shape[1]))
        sample = sample.type(torch.float32)
        return sample, target
        
        

def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    train_domain: str = None,
    semi_supervised: float = None,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        train_domain=train_domain,
    )

    if semi_supervised is not None:
        idxs = train_test_split(
            range(len(train_dataset)),
            train_size=semi_supervised,
            stratify=train_dataset.targets,
            random_state=42,
        )[0]
        train_dataset = Subset(train_dataset, idxs)
    
    print("VAL LOADER: -----------------", len(val_dataset.classes), ", classes ", len(val_dataset.samples), ", samples ", \
            len(val_dataset.targets), ", targets ", len(val_dataset.samples) / len(val_dataset.classes), ", samples per class")
    
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader

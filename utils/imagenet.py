# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import copy

# New imports for Albumentations addition
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def pil_loader(path):

    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))
    


#  Custom Κλάση για το ISIC DATASET - ALBUMENTATIONS
class ISICDataset(Dataset):
    def __init__(self, imagenet_folder, train, transform):
    
        self.folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        self.transform = transform
        
        
        self.samples = [
            os.path.join(self.folder, f) 
            for f in os.listdir(self.folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.folder}. Check your paths!")

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, index):
    #     img_path = self.samples[index]
        
    #     img = pil_loader(img_path)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img
    
    # New __getitem__ for Albumentations 
    def __getitem__(self, index):
        img_path = self.samples[index]
        image = pil_loader(img_path)
        
        # ΜΕΤΑΤΡΟΠΗ ΣΕ NUMPY ΓΙΑ ΤΟ ALBUMENTATIONS
        image = np.array(image) 
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image


# #  Custom Κλάση για το ISIC DATASET - TORCHVISION
# class ISICDataset(Dataset):
#     def __init__(self, imagenet_folder, train, transform):
    
#         self.folder = os.path.join(imagenet_folder, 'train' if train else 'val')
#         self.transform = transform
        
        
#         self.samples = [
#             os.path.join(self.folder, f) 
#             for f in os.listdir(self.folder) 
#             if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ]
        
#         if len(self.samples) == 0:
#             raise RuntimeError(f"Found 0 images in {self.folder}. Check your paths!")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         img_path = self.samples[index]
        
#         img = pil_loader(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img 

     



def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """

    # # Ίδια train transforms με το supervised pretraining
    # trans_train = transforms.Compose([
    #     #transforms.Resize((input_size, input_size), interpolation=interpolation),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=180),
    #     transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.00),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # trans_val = transforms.Compose([
    #     #transforms.Resize((input_size, input_size), interpolation=interpolation),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])




    # ============================New Albumentations transformations ============================
    # FROM Kaggle Challenge 2020 1st place solution (excluding CUTOUT, blur, elastic transform)
    trans_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        
        # A.OneOf([
        #     A.MotionBlur(blur_limit=5),
        #     A.GaussianBlur(blur_limit=5),
        #     A.GaussNoise(var_limit=(5.0, 30.0)),
        # ], p=0.5),

        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=0.2),
        #     A.GridDistortion(num_steps=5, distort_limit=0.2),
        # ], p=0.3),

        A.CLAHE(clip_limit=2.0, p=0.7), 
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        
        #A.Resize(input_size, input_size),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    trans_val = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


    # # Ιδια με το SparK github
    # trans_train = transforms.Compose([
    #     transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # trans_val = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])



    
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]
    
    full_dataset = ISICDataset(imagenet_folder=dataset_path, train=True, transform=None)

    indices = list(range(len(full_dataset)))
    print(f"LEN OF FULL DATASET: {len(full_dataset)}")
    train_indices, val_indices = train_test_split(
        indices,
        test_size = 5000,
        random_state=42,
        shuffle=True
    )

    train_ds = Subset(copy.deepcopy(full_dataset), train_indices)
    train_ds.dataset.transform = trans_train

    val_ds = Subset(copy.deepcopy(full_dataset), val_indices)
    val_ds.dataset.transform = trans_val

    print(f"[Dataset] Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    #print_transform(trans_train, '[pre-train]')






    print("\n" + "="*30)
    print("VALIDATION SET FILENAMES")
    print("="*30)
    
    # Παίρνουμε τα paths από το original dataset χρησιμοποιώντας τα indices του validation
    # Τα ταξινομούμε κιόλας για να είναι εύκολη η σύγκριση αν χρειαστεί
    val_filenames = [full_dataset.samples[i] for i in val_indices]
    
    # Εκτύπωση των πρώτων 20 για γρήγορο έλεγχο
    for path in val_filenames[:20]:
        print(os.path.basename(path))
        
    print(f"... and {len(val_filenames) - 20} more.")
    print("="*30 + "\n")
    
    # (Προαιρετικά) Αποθήκευση σε αρχείο για πλήρη σύγκριση
    with open("val_set_debug.txt", "w") as f:
        for path in val_filenames:
            f.write(os.path.basename(path) + "\n")
    print("[Debug] All validation filenames saved to val_set_debug.txt")





    
    return train_ds, val_ds


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')

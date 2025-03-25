import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import shutil
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torch.utils.data import DataLoader
from utils import random_horizontal_flip, random_rotation, random_brightness_contrast, mixup, cutmix

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        image = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)

        return {
            "image": image,
            "mask": mask
        }
    
class AugmentedOxfordPetDataset(SimpleOxfordPetDataset):
    def __init__(self, root, mode="train", transform=None, use_mixup=False, use_cutmix=False):
        super().__init__(root, mode, transform)
        self.use_mixup = use_mixup if mode == "train" else False
        self.use_cutmix = use_cutmix if mode == "train" else False

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        image, mask = sample["image"], sample["mask"]

        if self.mode == "train":
            # Augmentation
            image, mask = random_horizontal_flip(image, mask)
            image, mask = random_rotation(image, mask)
            # image, mask = random_scaling(image, mask)
            image = random_brightness_contrast(image)

        image = np.array(image).astype(np.float32) / 255.0  # HWC
        mask = np.array(mask).astype(np.float32) # HW

        if self.mode == "train":
            # MixUp
            if self.use_mixup and random.random() < 0.5:
                mix_idx = random.randint(0, len(self.filenames) - 1)
                sample_mix = super().__getitem__(mix_idx)
                image_mix, mask_mix = np.array(sample_mix["image"]).astype(np.float32) / 255.0, np.array(sample_mix["mask"]).astype(np.float32) 
                image, mask = mixup(image, mask, image_mix, mask_mix)

            # CutMix
            if self.use_cutmix and random.random() < 0.5:
                cut_idx = random.randint(0, len(self.filenames) - 1)
                sample_cut = super().__getitem__(cut_idx)
                image_cut, mask_cut = np.array(sample_cut["image"]).astype(np.float32) / 255.0, np.array(sample_cut["mask"]).astype(np.float32) 
                image, mask = cutmix(image, mask, image_cut, mask_cut)

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            sample = {"image": image, "mask": mask}

        return sample
    
class ToTensorTransform:
    def __call__(self, image, mask):
        transform = T.Compose([ 
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        return {
            "image": transform(image),  # **HWC -> CHW**
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),  # HW -> 1HW
        }


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n

def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, batch_size, shuffle=True, num_workers=0, transform=None):
    dataset = AugmentedOxfordPetDataset(root=data_path, mode=mode, use_cutmix=(mode == "train"), use_mixup=(mode == "train"), transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    OxfordPetDataset.download('/home/whp/dlp')
import os
import torch
import torchvision.transforms as T
import shutil
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torch.utils.data import DataLoader
from utils import random_horizontal_flip, random_rotation, random_scaling, random_brightness_contrast, mixup, cutmix

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

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

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

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample
    
class AugmentedOxfordPetDataset(SimpleOxfordPetDataset):
    def __init__(self, root, mode="train", transform=None, use_mixup=False, use_cutmix=False):
        super().__init__(root, mode, transform)
        self.use_mixup = use_mixup if mode == "train" else False
        self.use_cutmix = use_cutmix if mode == "train" else False

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]

        image = Image.fromarray(np.moveaxis(image, 0, -1))
        mask = Image.fromarray(mask[0])
        trimap = Image.fromarray(trimap[0])

        if self.mode == "train":
            # Augmentation
            image, mask, trimap = random_horizontal_flip(image, mask, trimap)
            image, mask, trimap = random_rotation(image, mask, trimap)
            image, mask, trimap = random_scaling(image, mask, trimap)
            image = random_brightness_contrast(image)

            # MixUp
            if self.use_mixup and random.random() < 0.5:
                mix_idx = random.randint(0, len(self.filenames) - 1)
                sample_mix = super().__getitem__(mix_idx)
                image_mix, mask_mix, trimap_mix = sample_mix["image"], sample_mix["mask"], sample_mix["trimap"]
                image_mix = Image.fromarray(np.moveaxis(image_mix, 0, -1))
                mask_mix = Image.fromarray(mask_mix[0])
                trimap_mix = Image.fromarray(trimap_mix[0])
                image, mask, trimap = mixup(image, mask, trimap, image_mix, mask_mix, trimap_mix)

            # CutMix
            if self.use_cutmix and random.random() < 0.5:
                cut_idx = random.randint(0, len(self.filenames) - 1)
                sample_cut = super().__getitem__(cut_idx)
                image_cut, mask_cut, trimap_cut = sample_cut["image"], sample_cut["mask"], sample_cut["trimap"]
                image_cut = Image.fromarray(np.moveaxis(image_cut, 0, -1))
                mask_cut = Image.fromarray(mask_cut[0])
                trimap_cut = Image.fromarray(trimap_cut[0])
                image, mask, trimap = cutmix(image, mask, trimap, image_cut, mask_cut, trimap_cut)

        image = np.moveaxis(np.array(image), -1, 0)
        mask = np.expand_dims(np.array(mask), 0)
        trimap = np.expand_dims(np.array(trimap), 0)

        sample["image"] = image
        sample["mask"] = mask
        sample["trimap"] = trimap
        return sample

class ToTensorTransform:
    def __call__(self, image, mask, trimap):
        transform = T.Compose([ 
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        return {
            "image": transform(image),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
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

def load_dataset(data_path, mode, batch_size, shuffle=True, num_workers=0, transform=ToTensorTransform()):
    dataset = AugmentedOxfordPetDataset(root=data_path, mode=mode, use_cutmix=(mode == "train"), use_mixup=(mode == "train"), transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
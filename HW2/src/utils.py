import random
from PIL import Image, ImageEnhance
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

def dice_score(pred, target, smooth=1e-6):
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def random_horizontal_flip(image, mask, p=0.5):
    if random.random() < p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask

def random_rotation(image, mask, angle_range=(-30, 30)):
    angle = random.uniform(angle_range[0], angle_range[1])
    image = image.rotate(angle, Image.BILINEAR)
    mask = mask.rotate(angle, Image.NEAREST)
    return image, mask

# def random_scaling(image, mask, scale_range=(0.8, 1.2)):
#     scale = random.uniform(scale_range[0], scale_range[1])
#     w, h = image.size
#     new_w, new_h = int(w * scale), int(h * scale)
#     image = image.resize((new_w, new_h), Image.BILINEAR)
#     mask = mask.resize((new_w, new_h), Image.NEAREST)
#     return image, mask

def random_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
    contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return image

def mixup(image1, mask1, image2, mask2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    image = lam * image1 + (1 - lam) * image2
    mask = lam * mask1 + (1 - lam) * mask2  
    mask = (mask > 0.5).astype(np.float32)
    return image, mask

def cutmix(image1, mask1, image2, mask2):
    h, w = image1.shape[:2]
    cx, cy = np.random.randint(w), np.random.randint(h)
    cut_w, cut_h = w // 2, h // 2 
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    image = image1.copy()
    mask = mask1.copy()
    image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    return image, mask

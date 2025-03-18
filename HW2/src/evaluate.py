import torch
import numpy as np

def dice_score(pred, target, smooth=1e-6):
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate(net, data, device):
    net.eval()  
    dice_scores = []

    with torch.no_grad():
        for batch in data:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = net(images)
            preds = torch.sigmoid(outputs) > 0.5  

            dice = dice_score(preds, masks)
            dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)
    return mean_dice

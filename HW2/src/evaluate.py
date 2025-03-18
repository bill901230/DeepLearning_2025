import torch
import numpy as np
from utils import dice_score

def evaluate(net, data, device):
    net.eval()  
    dice_scores = []

    with torch.no_grad():
        for batch in data:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)
            # print(torch.unique(masks))
            outputs = net(images)
            # print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")

            preds = torch.sigmoid(outputs) > 0.5  



            dice = dice_score(preds, masks)
            dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)
    return mean_dice

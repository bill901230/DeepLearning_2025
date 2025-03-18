import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from evaluate import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weights')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--show', '-s', type=int, default=0, help='show results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "unet" in args.model.lower():
        print("unet")
        model = UNet().to(device)
    elif "resnet34" in args.model.lower():
        print("resnet34")
        model = ResNet34_UNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    test_loader = load_dataset(data_path=args.data_path, mode="valid", batch_size=args.batch_size, shuffle=False, num_workers=4)

    os.makedirs("predictions", exist_ok=True)
    os.makedirs("input_images", exist_ok=True)

    dice_scores = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].cpu().numpy()
            outputs = model(images)

            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(np.uint8) * 255

            if args.show:
                for j in range(images.size(0)):
                    image_np = images[j].cpu().numpy().transpose(1, 2, 0)
                    image_np = (image_np * 255).astype(np.uint8)
                    Image.fromarray(image_np).save(f"input_images/image_{i * args.batch_size + j}.png")

                    pred = preds[j, 0]
                    Image.fromarray(pred).save(f"predictions/pred_{i * args.batch_size + j}.png")

        dice_score = evaluate(model, test_loader, device)

    print(f"Average Dice Score: {dice_score:.4f}")
    
    

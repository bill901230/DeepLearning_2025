import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oxford_pet import SimpleOxfordPetDataset, load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from tqdm import tqdm


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = args.data_path
    if not os.path.exists(os.path.join(dataset_root, "images")):
        print("Dataset not found. Downloading now...")
        SimpleOxfordPetDataset.download(dataset_root)

    train_loader = load_dataset(data_path=dataset_root, mode="train", batch_size=args.batch_size, num_workers=4)
    valid_loader = load_dataset(data_path=dataset_root, mode="valid", batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Dataset loaded!")

    model = UNet().to(device)
    # model = ResNet34_UNet().to(device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_dice = 0.0
    print("start training...")

    for epoch in range(args.epochs):

        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")

        dice = evaluate(model, valid_loader, device)
        print(f"Validation Dice Score: {dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/best_unet_{args.epochs}_{args.batch_size}_{args.learning_rate}.pth")
            print("Model saved!")
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
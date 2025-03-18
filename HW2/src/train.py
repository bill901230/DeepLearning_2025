import argparse
import os
from oxford_pet import SimpleOxfordPetDataset, load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def train(args):
    # implement the training function here
    dataset_root = args.data_path
    if not os.path.exists(os.path.join(dataset_root, "images")):
        print("Dataset not found. Downloading now...")
        SimpleOxfordPetDataset.download(dataset_root)

    train_loader = load_dataset(data_path=dataset_root, mode="train", batch_size=args.batch_size)
    valid_loader = load_dataset(data_path=dataset_root, mode="valid", batch_size=args.batch_size, shuffle=False)

    # model = UNet()
    # model = ResNet34_UNet()
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
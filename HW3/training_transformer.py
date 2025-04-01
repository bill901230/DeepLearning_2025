import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.device = args.device

        self.args = args
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.best_val_loss = float('inf')
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Train epoch:{epoch}') as pbar:
            for i, imgs in enumerate(train_loader):
                imgs = imgs.to(self.device)
                logits, target = self.model(imgs)  #predict, gt
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    target.reshape(-1),
                )
                
                loss = loss / self.args.accum_grad
                loss.backward()
                
                if (i + 1) % self.args.accum_grad == 0 or (i + 1) == len(train_loader):
                    self.optim.step()
                    self.optim.zero_grad()
                
                total_loss += loss.item() * self.args.accum_grad
                pbar.set_postfix(loss=total_loss / (i + 1))
                pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validation') as pbar:
                for i, imgs in enumerate(val_loader):
                    imgs = imgs.to(self.device)
                    logits, target = self.model(imgs)
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        target.reshape(-1),
                    )
                    
                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / (i + 1))
                    pbar.update(1)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_path = os.path.join('transformer_checkpoints128', f'epoch_{epoch}.pt')
        best_model_path = os.path.join('transformer_checkpoints128', 'best_model.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        
        if is_best:
            transformer_state_dict = self.model.transformer.state_dict()
            torch.save(transformer_state_dict, best_model_path)
            print(f"Saved best model to {best_model_path}")
                    
        print(f"Saved checkpoint to {checkpoint_path}")
        

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.model.transformer.parameters(), 
        #     lr=1e-4, betas=(0.9, 0.96), 
        #     weight_decay=4.5e-2
        # )
        # scheduler = WarmupLinearLRSchedule(
        #     optimizer=self.optim,
        #     init_lr=1e-6,
        #     peak_lr=args.learning_rate,
        #     end_lr=0.,
        #     warmup_epochs=10,
        #     epochs=args.epochs,
        #     current_step=args.start_from_epoch
        # )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=1e-5
        )
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="lab3_dataset/train", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="lab3_dataset/val", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:5", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        
        val_loss = train_transformer.eval_one_epoch(val_loader)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
        
        train_transformer.scheduler.step()
        
        is_best = val_loss < train_transformer.best_val_loss
        if is_best:
            train_transformer.best_val_loss = val_loss
        
        if epoch % args.save_per_epoch == 0 or is_best:
            train_transformer.save_checkpoint(epoch, is_best)
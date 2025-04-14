#Trainer.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) 
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():  
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch
        self.beta = 0.0
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(self.current_epoch, 
                                                start=0.0, 
                                                stop=self.kl_anneal_ratio, 
                                                n_cycle=self.kl_anneal_cycle)
        elif self.kl_anneal_type == 'Monotonic':
            self.beta = min(self.kl_anneal_ratio, self.current_epoch / (self.kl_anneal_cycle * 0.5))
        else:  # 'w/o KL annealing'
            self.beta = 1.0

        
    def update(self):
        # 更新當前epoch和beta值
        self.current_epoch += 1

        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(self.current_epoch, 
                                                start=0.0, 
                                                stop=self.kl_anneal_ratio, 
                                                n_cycle=self.kl_anneal_cycle)
        elif self.kl_anneal_type == 'Monotonic':
            self.beta = min(1.0, self.current_epoch / (self.kl_anneal_cycle * 0.5))
        else:  # 'w/o KL annealing'
            self.beta = 1.0
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1):
        cycle_length = self.args.num_epoch // n_cycle
        cycle = n_iter // cycle_length
        pos = n_iter % cycle_length
        
        if pos < cycle_length // 2:
            value = start + (stop - start) * (pos / (cycle_length // 2))
        else:
            value = stop - (stop - start) * ((pos - cycle_length // 2) / (cycle_length // 2))
        
        return value
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        run_log_dir = os.path.join(args.log_dir, f"_{args.kl_anneal_type}_ep-{self.args.num_epoch}_bs-{self.args.batch_size}_klr-{args.kl_anneal_ratio}_sde-{self.tfr_sde}_dstep-{self.tfr_d_step:.2f}")
        self.writer = SummaryWriter(log_dir=run_log_dir)

        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        self._train_batch_idx = 0
        best_val_loss = float('inf')
        best_val_psnr = 0.0
        save_path = f"{self.args.save_root}_ep-{self.args.num_epoch}_bs-{self.args.batch_size}_klr-{args.kl_anneal_ratio}_sde-{self.tfr_sde}_dstep-{self.tfr_d_step:.2f}"

        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label)
                
                beta = self.kl_annealing.get_beta()
                self.tqdm_bar('train TeacherForcing: {:.1f}, beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
                self._train_batch_idx += 1

            current_val_loss, current_val_psnr = self.eval()

            if self.current_epoch % self.args.per_save == 0:
                ckpt_name = f"epoch={self.current_epoch}_loss-{current_val_loss:.4f}_psnr-{current_val_psnr:.4f}.ckpt"
                self.save(os.path.join(save_path, ckpt_name))

            print(f"Epoch {self.current_epoch}: Validation psnr = {current_val_psnr:.4f}, Best psnr = {best_val_psnr:.4f}")
            if current_val_psnr > best_val_psnr:
                best_val_psnr = current_val_psnr
                self.save(os.path.join(save_path, "best_checkpoint.ckpt"))
                print(f"Saved best checkpoint at epoch {self.current_epoch} with psnr {best_val_psnr:.4f}")

            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        self.writer.close()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0.0
        total_psnr = 0.0
        count = 0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            total_loss += loss.item()
            total_psnr += psnr.item()
            count += 1
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        avg_loss = total_loss / count if count > 0 else 0
        avg_psnr = total_psnr / count if count > 0 else 0
        self.writer.add_scalar('Loss/val_total', avg_loss, self.current_epoch)
        self.writer.add_scalar('Metrics/val_psnr', avg_psnr, self.current_epoch)
        self.writer.flush()
        return avg_loss, avg_psnr
    
    def training_one_step(self, img, label):
        self.optim.zero_grad()
    
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        seq_len = img.shape[0]
        batch_size = img.shape[1]
        
        kld_loss = 0
        mse_loss = 0
        
        prev_img_feat = self.frame_transformation(img[0])
        
        for i in range(1, seq_len):
            current_label_feat = self.label_transformation(label[i])

            z, mu, logvar = self.Gaussian_Predictor(prev_img_feat, current_label_feat)
            
            fusion_feat = self.Decoder_Fusion(prev_img_feat, current_label_feat, z)
            pred_frame = self.Generator(fusion_feat)
            
            mse_loss += self.mse_criterion(pred_frame, img[i])
            current_kld = kl_criterion(mu, logvar, batch_size)
            if torch.isnan(current_kld) or torch.isinf(current_kld):
                print("Warning: current_kld is NaN/Inf (value = {}). Force setting to 0.5.".format(current_kld.item()))
                current_kld = torch.tensor(0.5, device=current_kld.device, dtype=current_kld.dtype)


            kld_loss += current_kld

            if random.random() < self.tfr:
                prev_img_feat = self.frame_transformation(img[i])
            else:
                prev_img_feat = self.frame_transformation(pred_frame)
        
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kld_loss

        step = self._train_batch_idx
        self.writer.add_scalar('Loss/train_total', loss.item(), step)
        self.writer.add_scalar('Loss/train_mse', mse_loss.item(), step)
        self.writer.add_scalar('Loss/train_kld', kld_loss.item(), step)
        self.writer.add_scalar('Hyperparameters/beta', beta, step)
        self.writer.add_scalar('Hyperparameters/tfr', self.tfr, step)
        
        loss.backward()
        self.optimizer_step()
    
        return loss
        
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        seq_len = img.shape[0]
        batch_size = img.shape[1]
        
        kld_loss = 0
        mse_loss = 0
        psnr = 0
        
        decoded_frame_list = [img[0][0].cpu()]
        gt_frame_list = [img[0][0].cpu()]
        
        prev_img_feat = self.frame_transformation(img[0])
        
        for i in range(1, seq_len):
            current_label_feat = self.label_transformation(label[i])
            z, mu, logvar = self.Gaussian_Predictor(prev_img_feat, current_label_feat)
            
            fusion_feat = self.Decoder_Fusion(prev_img_feat, current_label_feat, z)
            pred_frame = self.Generator(fusion_feat)
            
            mse_loss += self.mse_criterion(pred_frame, img[i])
            kld_loss += kl_criterion(mu, logvar, batch_size)/(self.args.frame_W*self.args.frame_H)
            
            psnr += Generate_PSNR(pred_frame, img[i])
            
            decoded_frame_list.append(pred_frame[0].cpu())
            gt_frame_list.append(img[i][0].cpu())
            
            prev_img_feat = self.frame_transformation(pred_frame)
        
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kld_loss
        psnr = psnr / (seq_len - 1)
        
        if self.args.store_visualization:
            self.make_gif(decoded_frame_list, os.path.join(self.args.save_root, f'epoch={self.current_epoch}_gen.gif'))
            self.make_gif(gt_frame_list, os.path.join(self.args.save_root, f'epoch={self.current_epoch}_gt.gif'))
        
        return loss, psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    save_path = f"{args.save_root}_ep-{args.num_epoch}_bs-{args.batch_size}_klr-{args.kl_anneal_ratio}_sde-{args.tfr_sde}_dstep-{args.tfr_d_step:.2f}"
    os.makedirs(save_path, exist_ok=True)
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=12)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--cuda_device', type=int, default=0, help='Specific CUDA device number to use')
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")

    parser.add_argument('--log_dir', type=str, default='./tensorboard', help="Directory to save tensorboard logs")

    args = parser.parse_args()
    
    main(args)

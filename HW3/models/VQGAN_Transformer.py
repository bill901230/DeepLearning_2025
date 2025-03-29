import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        batch_size = x.size(0)
        _, z_indices, _ = self.vqgan.encode(x)
        z_indices = z_indices.view(batch_size, self.num_image_tokens)
        return z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear_gamma(ratio):
            return ratio
        def cosine_gamma(ratio):
            return math.cos(ratio * math.pi / 2)
        def square_gamma(ratio):
            return ratio ** 2
        
        if mode == "linear":
            return linear_gamma
        elif mode == "cosine":
            return cosine_gamma
        elif mode == "square":
            return square_gamma
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        batch_size = x.size(0)
        # print("Input shape:", x.shape)
        z_indices=self.encode_to_z(x) #ground truth
        
        # print("z_indices shape:", z_indices.shape) 

        # decide number of token to be masked
        mask_ratio = torch.rand(1).item()
        num_masked = int(self.num_image_tokens * self.gamma(mask_ratio))

        # mask
        mask = torch.zeros_like(z_indices, dtype=torch.bool)

        for i in range(batch_size):
            mask_indices = torch.randperm(self.num_image_tokens, device=z_indices.device)[:num_masked]
            for idx in mask_indices:
                mask[i, idx] = True

        z_indices_masked = z_indices.clone()
        z_indices_masked[mask] = self.mask_token_id

        # print("z_indices_masked shape:", z_indices_masked.shape)

        logits = self.transformer(z_indices_masked)  #transformer predict the probability of tokens
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices_predict=None, mask_bc=None, ratio=0.0):
        if z_indices_predict is None:
            z_indices_predict = torch.full(
                (1, self.num_image_tokens),
                self.mask_token_id,
                dtype=torch.long,
                device=self.transformer.device
            )
        if mask_bc is None:
            mask_bc = torch.ones_like(z_indices_predict, dtype=torch.bool)
        logits = self.transformer(z_indices_predict)

        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict_candidate = torch.max(logits, dim=-1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence = torch.where(
            mask_bc,
            confidence,
            torch.tensor(float('-inf'), device=confidence.device)
        )
        #sort the confidence for the rank 
        _, sorted_indices = torch.sort(confidence, descending=True)

        #define how much the iteration remain predicted tokens by mask scheduling
        num_to_keep = int(self.num_image_tokens * (1 - self.gamma(ratio)))

        new_z_indices_predict = z_indices_predict.clone()
        new_mask_bc = mask_bc.clone()

        ##At the end of the decoding process, add back the original(non-masked) token values
        for idx in sorted_indices[0][:num_to_keep]:
            new_z_indices_predict[0, idx] = z_indices_predict_candidate[0, idx]
            new_mask_bc[0, idx] = False
        
        mask_bc=None
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        

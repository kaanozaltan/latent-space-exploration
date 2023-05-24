# Modified from https://github.com/adamian98/pulse

import os 

import torch
from torch import nn
import numpy as np

from stylegan import G_mapping, G_synthesis
from optimization import SphericalOptimizer
from loss import LossBuilder


class PULSE(nn.Module):
    def __init__(self):
        super(PULSE, self).__init__()

        # mapping only necessary if gaussian_fit.pt does not exist
        # self.mapping = G_mapping().cuda
        self.synthesis = G_synthesis().cuda()

        # with open('../models/mapping.pt', 'r') as f:
        #     mapping.load_state_dict(torch.load(f))

        with open('../models/synthesis.pt', 'rb') as f:
            self.synthesis.load_state_dict(torch.load(f))

        for param in self.synthesis.parameters():
            param.requires_grad = False
        
        if os.path.exists("../models/gaussian_fit.pt"):
            self.gaussian_fit = torch.load("../models/gaussian_fit.pt")
        else:
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000,512), dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(self.mapping(latent))
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit, "../models/gaussian_fit.pt")

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, ref_im, save_intermediate):
        loss_str = '100*L2+0.05*GEOCROSS'
        eps = 2e-3
        # can add noise_type
        # noise_type may require num_trainable_noise_layers
        # can add tile_latent (bool)
        # noise_type may also require bad_noise_layers
        # can use opt_name to pick opt_func from a dict
        learning_rate = 0.4
        steps = 100
        # can use lr_schedule to pick schedule_func from a dict
        # kwargs are given but not used

        batch_size = ref_im.shape[0]
        latent = torch.randn((batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')
        noise, noise_vars = [], []

        for i in range(18):
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            new_noise.requires_grad = False
            noise.append(new_noise)
        
        var_list = [latent]+noise_vars
        opt_func = torch.optim.Adam
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)
        schedule_func = lambda x: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        gen_im = None

        for j in range(steps):
            opt.opt.zero_grad()     
            latent_in = latent

            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])
            gen_im = (self.synthesis(latent_in, noise)+1)/2

            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            if(loss < min_loss):
                min_loss = loss
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']

            if(loss_l2 < min_l2):
                min_l2 = loss_l2

            if(save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            loss.backward()
            opt.step()
            scheduler.step()

            # edit
            # direction = np.load('superresolution/directions/smile.npy')
            # direction = np.repeat(direction, 18, axis=0)
            # direction = np.expand_dims(direction, axis=0)
            # direction = torch.from_numpy(direction).to(torch.float32).to('cuda')
            # alpha = 4

            # if j == steps - 1:
            #     latent_in = latent_in + alpha * direction
            #     gen_im = (self.synthesis(latent_in, noise)+1)/2

        yield (gen_im.clone().cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))

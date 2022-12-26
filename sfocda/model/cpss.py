import random
import torch
import torch.nn as nn

class CPSS(nn.Module):

    def __init__(self, p=0.3, num_h=2, num_w=2, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          num_h (int): height splited number.
          num_w (int): width splited number.          
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.num_h = num_h
        self.num_w = num_w

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu_all = []
        sig_all = []
        x_normed_all = []
        x_denormed_all = []
        x_cat_w = []
        H, W = x.size(2), x.size(3)

        ## Separate feature maps into patches and calculate patch styles
        for i in range(self.num_h):
            for j in range(self.num_w):
                x_patch = x[:, :, int(i*(H/self.num_h)):int((i+1)*(H/self.num_h)), int(j*(W/self.num_w)):int((j+1)*(W/self.num_w))]
                mu = x_patch.mean(dim=[2, 3], keepdim=True)
                sig = (x_patch.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
                mu, sig = mu.detach(), sig.detach()
                x_patch_normed = (x_patch - mu) / sig
                mu_all.append(mu)
                sig_all.append(sig)
                x_normed_all.append(x_patch_normed)

        ## Gather and shuffle patch style features
        mu_all = torch.cat(mu_all, dim=0)
        sig_all = torch.cat(sig_all, dim=0)
        perm = torch.randperm(B*self.num_h*self.num_w)
        mu_mix, sig_mix = mu_all[perm], sig_all[perm]

        ## De-normalize patches
        for p in range(len(x_normed_all)):
            x_denormed = x_normed_all[p] * sig_mix[p*B:(p+1)*B] + mu_mix[p*B:(p+1)*B]
            x_denormed_all.append(x_denormed)

        ## Generate feature maps from patches
        for k in range(self.num_h):
            x_cat_w.append(torch.cat(x_denormed_all[self.num_w*k:self.num_w*(k+1)], dim=3))

        x_stylized = torch.cat(x_cat_w, dim=2)

        return x_stylized

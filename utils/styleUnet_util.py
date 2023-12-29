import math
import random

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from model.op import conv2d_gradfix


class styleUnet_args(nn.Module):
    def __init__(self):
        super().__init__()
        ## Training
        self.iter = 800000  # total training iterations
        # n_sample = 64   # number of the samples generated during training
        self.latent = 64
        self.n_mlp = 4
        self.channel_multiplier = 2  # channel multiplier factor for the model. config-f = 2, else = 1
        self.start_iter = 0
        self.batch = 2  # batch sizes for each gpus   # <=4 or 4 * N
        self.wandb = True  # use weights and biases logging
        self.lr = 0.0005  # learning rate

        self.mixing = 0.9  # probability of latent code mixing

        self.augment = True  # apply non leaking augmentation
        self.augment_p = 0.  # probability of applying augmentation. 0 = use adaptive augmentation
        self.ada_target = 0.6  # target augmentation probability for adaptive augmentation
        self.ada_length = 500 * 1000  # target duraing to reach augmentation probability for adaptive augmentation
        self.ada_every = 256  # probability update interval of the adaptive augmentation

        self.path_regularize = 2.  # weight of the path length regularization
        self.path_batch_shrink = 2  # batch size reducing factor for the path length regularization (reduce memory consumption)
        self.g_reg_every = 4  # interval of the applying path length regularization
        self.view_dis_every = 0  ############################################ zxc: round view Dis
        self.r1 = 10.  # weight of the r1 regularization
        self.d_reg_every = 16  # interval of the applying r1 regularization


def requires_grad(model, flag=True):
    if type(model) == torch.nn.parameter.Parameter:
        model.requires_grad = flag
    elif type(model) == list:
        for p in model:
            p.requires_grad = flag
    else:
        for p in model.parameters():
            p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]

import torch
import torch.nn as nn
import math


class Embedder:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        out_dim += d * 2 * self.kwargs['num_freqs']
        self.max_freq = self.kwargs['max_freq_log2']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., self.max_freq, steps=self.N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)
        self.freq_bands = self.freq_bands.reshape(self.N_freqs, 1).to(self.kwargs['device'])
        self.out_dim = out_dim

    def embed(self, x, alpha=None):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""

        shape = list(x.shape[:-1]) + [-1]

        if self.N_freqs == 0:
            return x

        x_expanded = x.unsqueeze(-2)  # (..., 1, C).
        # Will be broadcasted to shape (F, C).
        angles = x_expanded * self.freq_bands  # [..., F, C]

        # The shape of the features is (F, 2, C) so that when we reshape it
        # it matches the ordering of the original NeRF code.
        # Vectorize the computation of the high-frequency (sin, cos) terms.
        # We use the trigonometric identity: cos(x) = sin(x + pi/2)
        features = torch.stack((angles, angles + math.pi / 2), dim=-2)  # [..., F, 2, C]
        # features = features.flatten()
        features = torch.sin(features)

        if alpha is not None:
            alpha = torch.clip(alpha - self.freq_bands, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(math.pi * alpha + math.pi))
            features = window.reshape(-1, 1, 1) * features  # [..., F, 2, C]
        features = features.reshape(shape)  # [..., F*2*C]
        # Prepend the original signal for the identity.
        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features

class Embedder_ori:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        self.max_freq = self.kwargs['max_freq_log2']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3, include_input=True, device='cuda'):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'device': device
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, alpha=None, eo=embedder_obj: eo.embed(x, alpha)
    return embed, embedder_obj.out_dim


def cosine_easing_window(num_bands, alpha, min_freq_log2=0, max_freq_log2=None):
    """Eases in each frequency one by one with a cosine.
    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.
    Args:
      min_freq_log2: the lower frequency band.
      max_freq_log2: the upper frequency band.
      num_bands: the number of frequencies.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.
    Returns:
      A 1-d numpy array with num_sample elements containing the window.
    """
    if max_freq_log2 is None:
      max_freq_log2 = num_bands - 1.0
    bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
    x = torch.clip(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * x + math.pi))
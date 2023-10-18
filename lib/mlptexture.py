# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import tinycudann as tcnn
import numpy as np


class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers'] - 1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()

        self.net.apply(self._init_weights)

        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale,))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


class MLPTexture3D(torch.nn.Module):
    def __init__(self, channels=3, internal_dims=32, hidden=2):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels - 1))

        enc_cfg = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale": per_level_scale
        }

        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        # self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling,))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims": self.encoder.n_output_dims,
            "n_output_dims": self.channels,
            "n_hidden_layers": hidden,
            "n_neurons": self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    # Sample texture at a given location
    def sample(self, texc):
        p_enc = self.encoder(texc)
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out)

        return out.view(*texc.shape[:-1], self.channels)  # Remap to [n, h, w, c]

    def cleanup(self):
        tcnn.free_temporary_memory()


if __name__ == '__main__':
    model = MLPTexture3D()
    x = torch.rand(1000,  3)
    out = model.sample(x)
    print(out.shape)
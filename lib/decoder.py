from typing import Tuple, Union
import torch
import torch.nn as nn
from diffusers.models.vae import Decoder


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (512,),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        return dec


if __name__ == '__main__':
    model = AutoencoderKL()
    z = torch.rand(1, 4, 64, 64)
    out = model(z)
    print(out.shape)


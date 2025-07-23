import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Backbone 3D VAE model with self-attention blocks for encoding and decoding scans for InstantGroup.
    """
    def __init__(self,
                 im_channels=1,
                 down_channels=[16, 32, 32, 32],
                 mid_channels=[32, 32],
                 down_sample=[True, True, True],
                 num_down_layers=2,
                 num_mid_layers=2,
                 num_up_layers=2,
                 z_channels=1,
                 norm_groups=8,
                 num_heads=4):
        """
        Parameters:
            im_channels: Number of input image channels.
            down_channels: Number of features in encoder (down-sampling) blocks (DownBlcok).
            mid_channels: Number of features in bottleneck blocks (MidBlock).
            down_sample: Whether to apply down-sampling for each encoder (down-sampling) blocks.
            num_down_layers: Number of convolutional layers of each DownBlcok.
            num_mid_layers: Number of convolutional layers of each MidBlock.
            num_up_layers: Number of convolutional layers of each UpBlock.
            z_channels: Number of channels (dimensionality) of the latent vector.
            norm_groups: Number of groups to separate the channels into for GroupNorm.
            num_heads: Number of attention heads in the MidBlock.
        """

        super().__init__()
        self.im_channels = im_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.down_sample = down_sample
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers

        # Latent Dimension
        self.z_channels = z_channels
        self.norm_groups = norm_groups
        self.num_heads = num_heads

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # Encoder
        self.encoder_conv_in = nn.Conv3d(self.im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1, 1))

        # downblock + midblock
        self.encoder_downs = nn.ModuleList([])     # down + normalization + activation
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 down_sample=self.down_sample[i],
                                                 num_layers=self.num_down_layers,
                                                 norm_groups=self.norm_groups))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_groups=self.norm_groups))

        self.encoder_out = nn.Sequential(
            nn.GroupNorm(self.norm_groups, self.down_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(self.down_channels[-1], 2 * self.z_channels, kernel_size=3, padding=1),
            nn.Conv3d(2 * self.z_channels, 2 * self.z_channels, kernel_size=1) # mean and variance concatenated
        )

        # Decoder
        self.up_sample = list(reversed(self.down_sample))

        self.decoder_in = nn.Sequential(
            nn.Conv3d(self.z_channels, self.z_channels, kernel_size=1),
            nn.Conv3d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1, 1))
        )

        # midblock + upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_groups=self.norm_groups))

        self.decoder_ups = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_ups.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               up_sample=self.down_sample[i - 1],
                                               num_layers=self.num_up_layers,
                                               norm_groups=self.norm_groups))

        self.decoder_out = nn.Sequential(
            nn.GroupNorm(self.norm_groups, self.down_channels[0]),
            nn.SiLU(),
            nn.Conv3d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
        )

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_downs):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)

        out = self.encoder_out(out)
        mean, logvar = torch.chunk(out, chunks=2, dim=1)
        sample = self.sample_z(mean, logvar)
        return sample, out

    def decode(self, z):
        out = self.decoder_in(z)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_ups:
            out = up(out)

        out = self.decoder_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output


class ResNetBlock(nn.Module):
    """
    Residual Convolutional Unit for DownBlcok, MidBlock and UpBlock
    """
    def __init__(self, in_channels, out_channels, norm_groups):
        super().__init__()
        self.res_conv_net = nn.Sequential(
            nn.GroupNorm(norm_groups, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.res_conv_net(x) + self.skip(x)


class DownBlock(nn.Module):
    """
    The encoder part of VAE with down-sampling.
    """
    def __init__(self, in_channels, out_channels,
                 down_sample, num_layers, norm_groups):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.resnet = nn.ModuleList([
            ResNetBlock(in_channels if i == 0 else out_channels, out_channels, norm_groups)
            for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) \
            if self.down_sample else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # Consecutive Resnet block
            out = self.resnet[i](out)
        # Down-sampling
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    """
    Bottleneck resnet convolutional block with attention layers.
    """
    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_groups):
        super().__init__()
        self.num_layers = num_layers

        self.resnet = nn.ModuleList([
            ResNetBlock(in_channels if i == 0 else out_channels, out_channels, norm_groups)
            for i in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_groups, out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        # Resnet block
        out = self.resnet[0](x)

        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w, d = out.shape
            in_attn = out.reshape(batch_size, channels, h * w * d)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w, d)
            out = out + out_attn

            # Resnet Block
            out = self.resnet[i+1](out)

        return out


class UpBlock(nn.Module):
    """
    Upsampling resnet convolutional block.
    """
    def __init__(self, in_channels, out_channels,
                 up_sample, num_layers, norm_groups):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        
        self.resnet = nn.ModuleList([
            ResNetBlock(in_channels if i == 0 else out_channels, out_channels, norm_groups)
            for i in range(num_layers)
        ])

        self.up_sample_conv = nn.ConvTranspose3d(in_channels, in_channels,
                                                 kernel_size=4, stride=2, padding=1) \
            if self.up_sample else nn.Identity()


    def forward(self, x):
        # Upsampling
        out = self.up_sample_conv(x)
        for i in range(self.num_layers):
            # Resnet Block
            out = self.resnet[i](out)

        return out

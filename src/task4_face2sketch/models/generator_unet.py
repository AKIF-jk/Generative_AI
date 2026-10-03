import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_styles=3, style_dim=16, dropout=0.0):
        super().__init__()
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        
        # We concatenate style embedding to the input image, so in_channels increases by style_dim
        # Actually, if we broadcast it spatially, it adds `style_dim` channels.
        total_in_channels = in_channels + style_dim
        
        self.down1 = UNetDown(total_in_channels, base_channels, normalize=False)
        self.down2 = UNetDown(base_channels, base_channels * 2)
        self.down3 = UNetDown(base_channels * 2, base_channels * 4)
        self.down4 = UNetDown(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down5 = UNetDown(base_channels * 8, base_channels * 8, dropout=dropout)
        self.down6 = UNetDown(base_channels * 8, base_channels * 8, dropout=dropout)
        self.down7 = UNetDown(base_channels * 8, base_channels * 8, normalize=False)

        self.up1 = UNetUp(base_channels * 8, base_channels * 8, dropout=dropout)
        self.up2 = UNetUp(base_channels * 16, base_channels * 8, dropout=dropout)
        self.up3 = UNetUp(base_channels * 16, base_channels * 8, dropout=dropout)
        self.up4 = UNetUp(base_channels * 16, base_channels * 4)
        self.up5 = UNetUp(base_channels * 8, base_channels * 2)
        self.up6 = UNetUp(base_channels * 4, base_channels)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, img, style_id):
        # img: (B, C, H, W)
        # style_id: (B,)
        batch_size, _, height, width = img.shape
        style_emb = self.style_embedding(style_id) # (B, style_dim)
        # Broadcast spatially
        style_emb_spatial = style_emb.view(batch_size, -1, 1, 1).expand(-1, -1, height, width) # (B, style_dim, H, W)
        
        x = torch.cat((img, style_emb_spatial), dim=1) # (B, C + style_dim, H, W)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


def infer_arch_from_state_dict(state_dict):
    """Recover (num_styles, style_dim, base_channels) from a GeneratorUNet state dict."""
    if 'down1.model.0.weight' not in state_dict:
        raise KeyError("Not a GeneratorUNet state dict: missing 'down1.model.0.weight'")
    first_conv = state_dict['down1.model.0.weight']
    base_channels = int(first_conv.shape[0])
    style_dim = int(first_conv.shape[1]) - 3
    num_styles = int(state_dict['style_embedding.weight'].shape[0])
    return num_styles, style_dim, base_channels

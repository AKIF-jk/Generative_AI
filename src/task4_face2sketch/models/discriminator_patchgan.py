import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_styles=3, style_dim=16):
        """
        in_channels is the number of channels of the image (photo + sketch) = 3 + 3 = 6
        But wait, style embedding is also concatenated. 
        So total in_channels to the first conv = 3 (photo) + 3 (sketch) + style_dim
        """
        super().__init__()
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        
        total_in_channels = in_channels * 2 + style_dim # Photo (3) + Sketch (3) + Style
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(total_in_channels, base_channels, normalization=False),
            *discriminator_block(base_channels, base_channels * 2),
            *discriminator_block(base_channels * 2, base_channels * 4),
            # Add one more conv layer with stride 1
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Final output layer
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, photo, sketch, style_id):
        # photo: (B, 3, H, W)
        # sketch: (B, 3, H, W)
        batch_size, _, height, width = photo.shape
        style_emb = self.style_embedding(style_id)
        style_emb_spatial = style_emb.view(batch_size, -1, 1, 1).expand(-1, -1, height, width)
        
        # Concatenate photo, sketch, and style embedding along channel dimension
        x = torch.cat((photo, sketch, style_emb_spatial), dim=1)
        
        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, target_is_real):
        # Create target tensor of the same shape as logits
        if target_is_real:
            target = torch.ones_like(logits)
        else:
            target = torch.zeros_like(logits)
        return self.loss_fn(logits, target)

_default_gan_loss = GANLoss()

def compute_generator_loss(disc_logits_fake, generated_sketch, real_sketch, lambda_l1=100.0, criterion_gan=None):
    if criterion_gan is None:
        criterion_gan = _default_gan_loss
    gan_loss = criterion_gan(disc_logits_fake, True)
    l1_loss = F.l1_loss(generated_sketch, real_sketch)
    total_loss = gan_loss + lambda_l1 * l1_loss
    return total_loss, gan_loss, l1_loss

def compute_discriminator_loss(disc_logits_real, disc_logits_fake, criterion_gan=None):
    if criterion_gan is None:
        criterion_gan = _default_gan_loss
    loss_real = criterion_gan(disc_logits_real, True)
    loss_fake = criterion_gan(disc_logits_fake, False)
    total_loss = (loss_real + loss_fake) * 0.5
    return total_loss, loss_real, loss_fake

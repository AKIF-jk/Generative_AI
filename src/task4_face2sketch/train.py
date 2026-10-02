import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import yaml
import numpy as np
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split
import json

from data.dataset import FS2KPairedDataset
from data.augmentations import PairedTransform, PairedValTransform
from models.generator_unet import GeneratorUNet
from models.discriminator_patchgan import DiscriminatorPatchGAN
from losses import compute_generator_loss, compute_discriminator_loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def get_dataloaders(data_dir, batch_size=16):
    set_seed(42)
    anno_file = os.path.join(data_dir, 'anno_train.json')
    with open(anno_file, 'r') as f:
        train_anno = json.load(f)
        
    labels = [x['style'] for x in train_anno]
    indices = np.arange(len(train_anno))
    
    # 85/15 stratified split based on sketch styles
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    
    full_train_dataset = FS2KPairedDataset(data_dir, split='train', transform=PairedTransform())
    full_val_dataset = FS2KPairedDataset(data_dir, split='train', transform=PairedValTransform())
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train_one_epoch(gen, disc, dataloader, opt_g, opt_d, device, lambda_l1=100.0):
    gen.train()
    disc.train()
    
    epoch_metrics = {'d_real_loss': 0.0, 'd_fake_loss': 0.0, 'g_adv_loss': 0.0, 'g_l1_loss': 0.0}
    
    for photos, sketches, styles in tqdm(dataloader, desc="Training", leave=False):
        photos = photos.to(device)
        sketches = sketches.to(device)
        styles = styles.to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        
        # Real loss
        real_logits = disc(photos, sketches, styles)
        
        # Fake loss
        fake_sketches = gen(photos, styles)
        fake_logits = disc(photos, fake_sketches.detach(), styles)
        
        d_loss, d_real_loss, d_fake_loss = compute_discriminator_loss(real_logits, fake_logits)
        d_loss.backward()
        opt_d.step()
        
        # ---------------------
        #  Train Generator
        # ---------------------
        opt_g.zero_grad()
        
        fake_logits_for_g = disc(photos, fake_sketches, styles)
        
        g_loss, g_adv_loss, g_l1_loss = compute_generator_loss(fake_logits_for_g, fake_sketches, sketches, lambda_l1)
        g_loss.backward()
        opt_g.step()
        
        epoch_metrics['d_real_loss'] += d_real_loss.item()
        epoch_metrics['d_fake_loss'] += d_fake_loss.item()
        epoch_metrics['g_adv_loss'] += g_adv_loss.item()
        epoch_metrics['g_l1_loss'] += g_l1_loss.item()
        
    for k in epoch_metrics:
        epoch_metrics[k] /= max(1, len(dataloader))
        
    return epoch_metrics

@torch.no_grad()
def validate(gen, dataloader, device):
    gen.eval()
    style_losses = {0: [], 1: [], 2: []}
    all_l1 = []
    
    for photos, sketches, styles in tqdm(dataloader, desc="Validating", leave=False):
        photos = photos.to(device)
        sketches = sketches.to(device)
        styles = styles.to(device)
        
        fake_sketches = gen(photos, styles)
        
        for s in range(3):
            mask = (styles == s)
            if mask.any():
                l1_val = torch.nn.functional.l1_loss(fake_sketches[mask], sketches[mask]).item()
                style_losses[s].append(l1_val)
                
        all_l1.append(torch.nn.functional.l1_loss(fake_sketches, sketches).item())
        
    val_metrics = {
        'val_l1_loss': np.mean(all_l1) if all_l1 else 0.0,
        'val_l1_style1': np.mean(style_losses[0]) if style_losses[0] else 0.0,
        'val_l1_style2': np.mean(style_losses[1]) if style_losses[1] else 0.0,
        'val_l1_style3': np.mean(style_losses[2]) if style_losses[2] else 0.0,
    }
    return val_metrics

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    run = wandb.init(project="FS2K_Task4", config=config)
    cfg = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(cfg.data_dir, batch_size=cfg.batch_size)
    
    # Cache fixed validation images for consistent progression monitoring across epochs
    fixed_photos, fixed_sketches, fixed_styles = next(iter(val_loader))
    fixed_photos = fixed_photos[:4].to(device)
    fixed_sketches = fixed_sketches[:4].to(device)
    fixed_styles = fixed_styles[:4].to(device)
    
    gen = GeneratorUNet(base_channels=cfg.base_channels, style_dim=cfg.style_dim, dropout=cfg.dropout).to(device)
    disc = DiscriminatorPatchGAN(base_channels=cfg.base_channels, style_dim=cfg.style_dim).to(device)
    
    opt_g = optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
    
    # Optional linear decay scheduler after 50% epochs
    decay_start = cfg.epochs // 2
    lr_lambda = lambda epoch: 1.0 - max(0, epoch - decay_start) / max(1, (cfg.epochs - decay_start))
    scheduler_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_lambda)
    scheduler_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lr_lambda)
    
    best_val_l1 = float('inf')
    
    for epoch in range(1, cfg.epochs + 1):
        metrics = train_one_epoch(gen, disc, train_loader, opt_g, opt_d, device, lambda_l1=cfg.lambda_l1)
        val_metrics = validate(gen, val_loader, device)
        
        scheduler_g.step()
        scheduler_d.step()
        
        log_payload = {
            'epoch': epoch,
            'train_d_real_loss': metrics['d_real_loss'],
            'train_d_fake_loss': metrics['d_fake_loss'],
            'train_g_adv_loss': metrics['g_adv_loss'],
            'train_g_l1_loss': metrics['g_l1_loss'],
            'val_l1_loss': val_metrics['val_l1_loss'],
            'val_l1_style1': val_metrics['val_l1_style1'],
            'val_l1_style2': val_metrics['val_l1_style2'],
            'val_l1_style3': val_metrics['val_l1_style3'],
            'lr_g': opt_g.param_groups[0]['lr'],
            'lr_d': opt_d.param_groups[0]['lr']
        }
        
        print(f"Epoch {epoch:03d}/{cfg.epochs:03d} | Train G L1: {metrics['g_l1_loss']:.4f} | Val L1: {val_metrics['val_l1_loss']:.4f}")
        
        # Periodic sample image logging using identical validation photos
        if epoch % cfg.save_interval == 0 or epoch == 1:
            gen.eval()
            with torch.no_grad():
                gen_sketches = gen(fixed_photos, fixed_styles)
                # Denormalize [-1, 1] to [0, 1]
                vis_real_photos = (fixed_photos.cpu() * 0.5 + 0.5).clamp(0, 1)
                vis_real_sketches = (fixed_sketches.cpu() * 0.5 + 0.5).clamp(0, 1)
                vis_fake_sketches = (gen_sketches.cpu() * 0.5 + 0.5).clamp(0, 1)
                
                # Stack Photo, Real Sketch, Generated Sketch
                comparison = torch.cat([vis_real_photos, vis_real_sketches, vis_fake_sketches], dim=0)
                grid = vutils.make_grid(comparison, nrow=fixed_photos.size(0))
                log_payload['val_samples'] = wandb.Image(grid, caption=f"Epoch {epoch} (Photo / Real / Generated)")
                
            if run and hasattr(wandb, 'run') and wandb.run:
                ckpt_dir = wandb.run.dir
            else:
                ckpt_dir = "."
            torch.save(gen.state_dict(), os.path.join(ckpt_dir, f'gen_epoch_{epoch}.pth'))
            
        wandb.log(log_payload)
        
        if val_metrics['val_l1_loss'] < best_val_l1:
            best_val_l1 = val_metrics['val_l1_loss']
            torch.save(gen.state_dict(), 'best_generator.pth')
            print(f"--> Saved new best generator with Val L1: {best_val_l1:.4f}")
            
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/task4.yaml')
    args = parser.parse_args()
    main(args.config)

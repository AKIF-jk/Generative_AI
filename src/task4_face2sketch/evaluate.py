import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

from data.dataset import FS2KPairedDataset
from data.augmentations import PairedValTransform
from models.generator_unet import GeneratorUNet

def evaluate_and_plot(model_path, data_dir, output_dir, base_channels=96, dropout=0.23843508579472333, style_dim=16, max_visualizations=20):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}")

    val_dataset = FS2KPairedDataset(data_dir, split='test', transform=PairedValTransform())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    gen = GeneratorUNet(base_channels=base_channels, style_dim=style_dim, dropout=dropout).to(device)
    if os.path.exists(model_path):
        gen.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with randomly initialized model.")
        
    gen.eval()
    
    style_l1 = {0: [], 1: [], 2: []}
    style_ssim = {0: [], 1: [], 2: []}
    saved_count = 0
    
    with torch.no_grad():
        for i, (photos, sketches, styles) in enumerate(tqdm(val_loader, desc="Evaluating")):
            photos = photos.to(device)
            styles = styles.to(device)
            fake_sketches = gen(photos, styles)
            
            # Denormalize from [-1, 1] to [0, 1]
            photos_norm = (photos * 0.5 + 0.5).clamp(0, 1)
            sketches_norm = (sketches * 0.5 + 0.5).clamp(0, 1)
            fake_norm = (fake_sketches * 0.5 + 0.5).clamp(0, 1)
            
            photos_np = (photos_norm.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')
            sketches_np = (sketches_norm.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')
            fake_np = (fake_norm.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')
            
            for j in range(photos.size(0)):
                s = styles[j].item()
                # Compute sample L1
                l1_val = torch.nn.functional.l1_loss(fake_sketches[j], sketches[j].to(device)).item()
                style_l1[s].append(l1_val)
                
                # Compute sample SSIM
                real_img = sketches_np[j]
                fake_img = fake_np[j]
                sample_ssim = ssim_fn(real_img, fake_img, channel_axis=2)
                style_ssim[s].append(sample_ssim)
                
                # Save visual comparisons with absolute error maps
                if saved_count < max_visualizations:
                    diff = np.abs(sketches_norm[j].cpu().numpy().transpose(1, 2, 0) - 
                                  fake_norm[j].cpu().numpy().transpose(1, 2, 0))
                    error_map = np.mean(diff, axis=-1)
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(photos_np[j])
                    axes[0].set_title("Input Photo")
                    axes[0].axis('off')
                    
                    axes[1].imshow(sketches_np[j])
                    axes[1].set_title(f"Target Sketch (Style {s+1})")
                    axes[1].axis('off')
                    
                    axes[2].imshow(fake_np[j])
                    axes[2].set_title(f"Generated (SSIM: {sample_ssim:.3f})")
                    axes[2].axis('off')
                    
                    im_err = axes[3].imshow(error_map, cmap='inferno')
                    axes[3].set_title(f"Absolute Error Map (L1: {l1_val:.4f})")
                    axes[3].axis('off')
                    plt.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'eval_sample_{saved_count:03d}_style_{s+1}.png'), bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1
                    
    print("\n" + "="*50)
    print("TASK 4 QUANTITATIVE EVALUATION RESULTS (TEST SET)")
    print("="*50)
    all_l1 = []
    all_ssim = []
    for s in range(3):
        avg_l1 = np.mean(style_l1[s]) if style_l1[s] else 0.0
        avg_ssim = np.mean(style_ssim[s]) if style_ssim[s] else 0.0
        all_l1.extend(style_l1[s])
        all_ssim.extend(style_ssim[s])
        print(f"Style {s+1}: L1 Loss = {avg_l1:.4f} | SSIM = {avg_ssim:.4f} (N={len(style_l1[s])})")
        
    print("-"*50)
    print(f"Overall Test Set: L1 Loss = {np.mean(all_l1):.4f} | SSIM = {np.mean(all_ssim):.4f} (Total N={len(all_l1)})")
    print(f"Visual evaluations and error maps saved to: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    evaluate_and_plot("best_generator.pth", "data/FS2K", "evaluation_results")

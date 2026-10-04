import optuna
import torch
import torch.optim as optim
import yaml
import os

import json
import pandas as pd

from data.dataset import FS2KPairedDataset
from data.augmentations import PairedTransform, PairedValTransform
from models.generator_unet import GeneratorUNet
from models.discriminator_patchgan import DiscriminatorPatchGAN
from train import get_dataloaders, train_one_epoch, validate

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Suggest hyperparameters matching the assignment specification
    lr_g = trial.suggest_float('lr_g', 1e-5, 5e-4, log=True)
    lr_d = trial.suggest_float('lr_d', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    base_channels = trial.suggest_categorical('base_channels', [32, 48, 64, 96])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    style_dim = trial.suggest_categorical('style_dim', [4, 8, 16, 32])
    lambda_l1 = trial.suggest_float('lambda_l1', 10, 200, log=True)
    
    data_dir = "data/FS2K"
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    gen = GeneratorUNet(base_channels=base_channels, style_dim=style_dim, dropout=dropout).to(device)
    disc = DiscriminatorPatchGAN(base_channels=base_channels, style_dim=style_dim).to(device)
    
    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Reduced epoch budget for HPO as specified in the assignment
    epochs = 5 
    
    for epoch in range(epochs):
        train_one_epoch(gen, disc, train_loader, opt_g, opt_d, device, lambda_l1=lambda_l1)
        val_metrics = validate(gen, val_loader, device)
        val_l1 = val_metrics['val_l1_loss']
        
        trial.report(val_l1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return val_l1

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=15)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Val L1): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)

    with open("optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("Saved optuna_trials.csv and optuna_best_params.json")

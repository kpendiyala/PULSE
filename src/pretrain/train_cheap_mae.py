import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset
from src.utils import plot_mae_losses, plot_reconstructions
from src.modules.hinge_loss import AllPairsHingeLoss


def _choose_modalities(
    available: list[str],
    enable_dropout: bool,
    p_drop_eda: float,
    p_drop_cheap: float,
    force_drop_eda: bool = False,
) -> list[str]:
    """Stochastically drop EDA and/or cheap sensors while ensuring at least one cheap modality remains.

    - available: list of modality names present in models/batch
    - enable_dropout: if False, returns available unchanged (unless force_drop_eda)
    - p_drop_eda: probability to drop EDA if present
    - p_drop_cheap: independent probability per cheap modality (ecg, bvp, acc, temp)
    - force_drop_eda: if True, drop EDA deterministically
    """
    cheap_modalities = [m for m in ['ecg', 'bvp', 'acc', 'temp'] if m in available]
    has_eda = 'eda' in available

    kept = set(available)

    # Optionally drop EDA
    if has_eda:
        drop_eda = force_drop_eda or (enable_dropout and (torch.rand(()) < p_drop_eda))

        if drop_eda and 'eda' in kept:
            kept.remove('eda')

    # Optionally drop cheap modalities
    if enable_dropout and cheap_modalities:
        for m in cheap_modalities:
            drop_cheap = (enable_dropout and (torch.rand(()) < p_drop_cheap))
            if drop_cheap and m in kept:
                kept.remove(m)

    # Ensure at least one cheap modality remains
    if not any(m in kept for m in cheap_modalities):
        # Randomly keep one cheap modality from the originals
        if cheap_modalities:
            idx = int(torch.randint(low=0, high=len(cheap_modalities), size=(1,)).item())
            kept.add(cheap_modalities[idx])

    # Ensure at least two modalities total if possible
    if len(kept) < 2 and len(available) >= 2:
        candidates = [m for m in available if m not in kept]
        if candidates:
            idx = int(torch.randint(low=0, high=len(candidates), size=(1,)).item())
            kept.add(candidates[idx])

    # Final safeguard: ensure something remains
    if not kept:
        # Prefer to bring back one cheap; otherwise bring back eda if it existed
        if cheap_modalities:
            kept.add(cheap_modalities[0])
        elif has_eda:
            kept.add('eda')

    return [m for m in available if m in kept]

def setup_logging(run_name, output_path):
    """Configures logging to file and console."""
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

def save_checkpoint(epoch, models, optimizers, schedulers, best_val_loss, losses, path):
    """Saves the training state to a checkpoint file."""
    state = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'losses': losses
    }
    for name, model in models.items():
        state[f'{name}_model_state_dict'] = model.state_dict()
    for name, optimizer in optimizers.items():
        state[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
    for name, scheduler in schedulers.items():
        state[f'{name}_scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(path, models, optimizers, schedulers, device):
    """Loads the training state from a checkpoint file."""
    if not os.path.isfile(path):
        logging.warning(f"Checkpoint file not found at {path}. Starting from scratch.")
        return 0, float('inf'), []
        
    checkpoint = torch.load(path, map_location=device)
    for name, model in models.items():
        model.load_state_dict(checkpoint[f'{name}_model_state_dict'])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[f'{name}_optimizer_state_dict'])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(checkpoint[f'{name}_scheduler_state_dict'])
        
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    losses = checkpoint.get('losses', [])
    logging.info(f"Resumed from epoch {epoch}. Best validation loss was {best_val_loss:.4f}.")
    return epoch, best_val_loss, losses

def validate(models, dataloader, device, alignment_loss_fn, alignment_loss_weight, disentangling_loss_weight=0.0, use_curriculum=False, epoch=0, num_epochs=1, enable_modality_dropout: bool=False, p_drop_eda: float=0.0, p_drop_cheap: float=0.0, val_without_eda: bool=False, val_with_dropout: bool=False, mask_ratio: float=0.75):
    """Runs a validation loop and returns the average loss."""
    for model in models.values():
        model.eval()
    total_loss, total_recon_loss, total_align_loss, total_disent_loss = 0, 0, 0, 0

    if use_curriculum:
        ratio = (epoch + 1) / num_epochs
        align_weight = np.sin(ratio * np.pi / 2)
        recon_weight = np.cos(ratio * np.pi / 2)
    else:
        align_weight = alignment_loss_weight
        recon_weight = 1.0

    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            # Build signals dict from a possibly dropped subset of modalities
            modality_names = list(models.keys())
            if val_with_dropout:
                kept_mods = _choose_modalities(modality_names, True, p_drop_eda, p_drop_cheap)
            elif val_without_eda:
                kept_mods = [m for m in modality_names if m != 'eda']
            else:
                kept_mods = modality_names
            signals = {name: batch[name].to(device) for name in kept_mods}
            
            # Step 1: Get all embeddings from the encoders
            all_private_embs, all_shared_embs, all_masks, all_ids_restore = {}, {}, {}, {}
            for name, sig in signals.items():
                model = models[name]
                ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, mask_ratio, device)
                private, shared, mask = model.forward_encoder(sig, mask_ratio, ids_shuffle, ids_restore, ids_keep)
                all_private_embs[name] = private
                all_shared_embs[name] = shared # Use dict for shared embeddings
                all_masks[name] = mask
                all_ids_restore[name] = ids_restore

            # Step 2: Create the aligned shared embedding by averaging
            aligned_shared_emb = torch.stack(list(all_shared_embs.values())).mean(dim=0)
            
            # Step 3: Calculate reconstruction loss using the aligned shared embedding
            num_kept = max(1, len(signals))
            reconstruction_loss = 0.0
            for name, sig in signals.items():
                rec = models[name].forward_decoder(all_private_embs[name], aligned_shared_emb, all_ids_restore[name])
                reconstruction_loss += models[name].reconstruction_loss(sig, rec, all_masks[name])
            reconstruction_loss = reconstruction_loss / num_kept

            # Step 4: Calculate alignment loss using Hinge Loss (skip if < 2 modalities)
            # First, get a single vector representation for each signal by averaging patch embeddings
            all_shared_embs_mean = {k: v.mean(dim=1) for k, v in all_shared_embs.items()}
            if len(all_shared_embs_mean) >= 2:
                # Then, normalize these summary vectors before passing to loss function
                normalized_shared_embs = {k: nn.functional.normalize(v, p=2, dim=1) for k, v in all_shared_embs_mean.items()}
                loss_align = alignment_loss_fn(normalized_shared_embs)
            else:
                loss_align = torch.tensor(0.0, device=device)
            
            # Step 4b: Disentangling loss between private and shared embeddings (cosine similarity)
            # Use only the non-zero parts based on the encoder's fixed private mask to avoid trivial orthogonality
            all_private_embs_mean = {k: v.mean(dim=1) for k, v in all_private_embs.items()}  # [B, D]
            if len(all_private_embs_mean) > 0:
                per_mod_disent = []
                for k in all_private_embs_mean.keys():
                    private_mask = models[k].encoder.private_mask  # [D]
                    private_idx = (private_mask > 0.5).nonzero(as_tuple=True)[0]
                    shared_idx = (private_mask <= 0.5).nonzero(as_tuple=True)[0]

                    v_p = all_private_embs_mean[k][:, private_idx]
                    v_s = all_shared_embs_mean[k][:, shared_idx]
                    # Handle unequal dimensionalities
                    min_dim = min(v_p.shape[1], v_s.shape[1])
                    if min_dim == 0:
                        continue
                    v_p = v_p[:, :min_dim]
                    v_s = v_s[:, :min_dim]
                    v_p = nn.functional.normalize(v_p, p=2, dim=1)
                    v_s = nn.functional.normalize(v_s, p=2, dim=1)
                    cos_sim = (v_p * v_s).sum(dim=1)  # [B]
                    per_mod_disent.append((cos_sim.pow(2)).mean())
                loss_disent = torch.stack(per_mod_disent).mean() if len(per_mod_disent) > 0 else torch.tensor(0.0, device=device)
            else:
                loss_disent = torch.tensor(0.0, device=device)
            
            loss = recon_weight * reconstruction_loss + align_weight * loss_align + disentangling_loss_weight * loss_disent
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_align_loss += loss_align.item()
            total_disent_loss += loss_disent.item()

    avg_total_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_align_loss = total_align_loss / len(dataloader)
    avg_disent_loss = total_disent_loss / len(dataloader)
    
    return {'total': avg_total_loss, 'recon': avg_recon_loss, 'align': avg_align_loss, 'disent': avg_disent_loss}


def train(args):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)
    
    # --- Models, Optimizers, Schedulers ---
    if args.big_model:
        args.decoder_depth += 2
        args.decoder_mlp_ratio *= 2
        args.embed_dim = 1024+256
        args.decoder_embed_dim = 1024+256
        args.num_heads = 8
        args.decoder_num_heads = 8

    # Determine modalities to include
    if getattr(args, 'modalities', 'all') == 'all':
        modality_names = ['ecg', 'bvp', 'acc', 'temp']
        if getattr(args, 'include_eda', False):
            modality_names.append('eda')
    else:
        parsed = [m.strip().lower() for m in args.modalities.split(',') if m.strip()]
        allowed = {'ecg', 'bvp', 'acc', 'temp', 'eda'}
        invalid = [m for m in parsed if m not in allowed]
        if len(invalid) > 0:
            raise ValueError(f"Invalid modalities specified: {invalid}. Allowed: {sorted(list(allowed))}")
        modality_names = parsed
        if getattr(args, 'include_eda', False) and 'eda' not in modality_names:
            modality_names.append('eda')

    logging.info(f"Using modalities: {modality_names}")
    if len(modality_names) < 2:
        logging.warning("Only one modality selected. Alignment loss will be skipped in training and validation.")

    models = {}
    for name in modality_names:
        models[name] = CheapSensorMAE(
            modality_name=name,
            sig_len=args.signal_length,
            window_len=args.patch_window_len,
            private_mask_ratio=args.private_mask_ratio,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_num_heads,
            mlp_ratio=args.mlp_ratio,
            decoder_mlp_ratio=args.decoder_mlp_ratio
        ).to(device)
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=args.learning_rate) for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.lr_restart_epochs, T_mult=args.t_mult) for name, opt in optimizers.items()}

    # --- Load Checkpoint ---
    start_epoch, best_val_loss, losses = 0, float('inf'), []
    if args.resume_from:
        start_epoch, best_val_loss, losses = load_checkpoint(args.resume_from, models, optimizers, schedulers, device)
    else:
        logging.info("--- Starting New Training Run ---")
        logging.info(f"Run Name: {args.run_name}")
        logging.info("Configuration:")
        for key, value in vars(args).items():
            logging.info(f"  {key}: {value}")
        if args.enable_modality_dropout:
            logging.info(f"  Modality Dropout: {args.p_drop_eda} for EDA, {args.p_drop_cheap} for cheap modalities")
        if args.val_with_dropout:
            logging.info(f"  Validation with Dropout: {args.p_drop_eda} for EDA, {args.p_drop_cheap} for cheap modalities")
        
        # Log model architecture (they are all the same)
        any_name = next(iter(models.keys()))
        any_model = models[any_name]
        logging.info(f"\nModel Architecture (showing '{any_name}'):")
        logging.info(str(any_model))
        logging.info(f"Total parameters: {sum(p.numel() for p in any_model.parameters()):,}")

        #cg_model = models['ecg']
        #ogging.info("\nModel Architecture:")
        #ogging.info(str(ecg_model))
        #ogging.info(f"Total parameters: {sum(p.numel() for p in ecg_model.parameters()):,}")

        logging.info("-" * 50)

    # --- Data ---
    # Load the full training dataset for the fold
    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train')

    # Reduce dataset size based on percentage argument
    if args.dataset_percentage < 1.0:
        original_size = len(full_train_dataset)
        subset_size = int(original_size * args.dataset_percentage)
        generator = torch.Generator().manual_seed(args.seed)
        full_train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {args.dataset_percentage*100:.0f}% of the dataset: {len(full_train_dataset)} out of {original_size} windows.")

    # Split training data into train and validation sets (90/10)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    # Get a fixed batch from the validation set for visualization
    vis_batch = next(iter(val_dataloader))
    
    alignment_loss_fn = AllPairsHingeLoss(alpha=args.hinge_alpha, neg_sample_percent=args.hinge_neg_sample_percent)
    
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        # --- Training Loop ---
        for model in models.values():
            model.train()
        train_total_loss, train_recon_loss, train_align_loss, train_disent_loss = 0, 0, 0, 0
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar_train):
            # Build signals dict dynamically from available models/modalities with modality dropout
            modality_names = list(models.keys())
            kept_mods = _choose_modalities(
                modality_names,
                enable_dropout=args.enable_modality_dropout,
                p_drop_eda=args.p_drop_eda,
                p_drop_cheap=args.p_drop_cheap,
            )
            signals = {name: batch[name].to(device) for name in kept_mods}
            
            # Zero gradients
            for opt in optimizers.values():
                opt.zero_grad()
                
            # Step 1: Get all embeddings from the encoders
            all_private_embs, all_shared_embs, all_masks, all_ids_restore = {}, {}, {}, {}
            for name, sig in signals.items():
                model = models[name]
                ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, args.mask_ratio, device)
                private, shared, mask = model.forward_encoder(sig, args.mask_ratio, ids_shuffle, ids_restore, ids_keep)
                all_private_embs[name] = private
                all_shared_embs[name] = shared # Use dict for shared embeddings
                all_masks[name] = mask
                all_ids_restore[name] = ids_restore

            # Step 2: Create the aligned shared embedding by averaging
            aligned_shared_emb = torch.stack(list(all_shared_embs.values())).mean(dim=0)

            # Step 3: Calculate reconstruction loss using the aligned shared embedding
            num_kept = max(1, len(signals))
            reconstruction_loss = 0.0
            for name, sig in signals.items():
                rec = models[name].forward_decoder(all_private_embs[name], aligned_shared_emb, all_ids_restore[name])
                reconstruction_loss += models[name].reconstruction_loss(sig, rec, all_masks[name])
            reconstruction_loss = reconstruction_loss / num_kept

            # Step 4: Calculate alignment loss using Hinge Loss (skip if < 2 modalities)
            # First, get a single vector representation for each signal by averaging patch embeddings
            all_shared_embs_mean = {k: v.mean(dim=1) for k, v in all_shared_embs.items()}
            if len(all_shared_embs_mean) >= 2:
                # Then, normalize these summary vectors before passing to loss function
                normalized_shared_embs = {k: nn.functional.normalize(v, p=2, dim=1) for k, v in all_shared_embs_mean.items()}
                loss_align = alignment_loss_fn(normalized_shared_embs)
            else:
                loss_align = torch.tensor(0.0, device=device)
            
            # Step 5: Disentangling loss between private and shared embeddings (cosine similarity)
            # Compare only non-zero parts from the encoder's fixed private mask
            all_private_embs_mean = {k: v.mean(dim=1) for k, v in all_private_embs.items()}  # [B, D]
            per_mod_disent = []
            for k in all_private_embs_mean.keys():
                private_mask = models[k].encoder.private_mask  # [D]
                private_idx = (private_mask > 0.5).nonzero(as_tuple=True)[0]
                shared_idx = (private_mask <= 0.5).nonzero(as_tuple=True)[0]

                v_p = all_private_embs_mean[k][:, private_idx]
                v_s = all_shared_embs_mean[k][:, shared_idx]
                min_dim = min(v_p.shape[1], v_s.shape[1])
                if min_dim == 0:
                    continue
                v_p = v_p[:, :min_dim]
                v_s = v_s[:, :min_dim]
                v_p = nn.functional.normalize(v_p, p=2, dim=1)
                v_s = nn.functional.normalize(v_s, p=2, dim=1)
                cos_sim = (v_p * v_s).sum(dim=1)
                per_mod_disent.append((cos_sim.pow(2)).mean())
            loss_disent = torch.stack(per_mod_disent).mean() if len(per_mod_disent) > 0 else torch.tensor(0.0, device=device)

            # Step 6: Combine losses
            if args.use_curriculum:
                ratio = (epoch + 1) / args.num_epochs
                align_weight = np.sin(ratio * np.pi / 2)
                recon_weight = np.cos(ratio * np.pi / 2)
                loss = recon_weight * reconstruction_loss + align_weight * loss_align + args.disentangling_loss_weight * loss_disent
            else:
                loss = reconstruction_loss + args.alignment_loss_weight * loss_align + args.disentangling_loss_weight * loss_disent

            loss.backward()
            active = set(signals.keys())
            for name, opt in optimizers.items():
                if name in active:
                    opt.step()
            # Per-iteration scheduler step (CosineAnnealingWarmRestarts expects fractional epoch)
            frac_epoch = epoch + (batch_idx + 1) / max(1, len(train_dataloader))
            for sch in schedulers.values():
                sch.step(frac_epoch)
                
            train_total_loss += loss.item()
            train_recon_loss += reconstruction_loss.item()
            train_align_loss += loss_align.item()
            train_disent_loss += loss_disent.item()
            pbar_train.set_postfix(Loss=loss.item())

        avg_train_total_loss = train_total_loss / len(train_dataloader)
        avg_train_recon_loss = train_recon_loss / len(train_dataloader)
        avg_train_align_loss = train_align_loss / len(train_dataloader)
        avg_train_disent_loss = train_disent_loss / len(train_dataloader)
        
        # --- Validation Loop ---
        val_losses = validate(models, val_dataloader, device, alignment_loss_fn, args.alignment_loss_weight, args.disentangling_loss_weight,
                              use_curriculum=args.use_curriculum, epoch=epoch, num_epochs=args.num_epochs,
                              enable_modality_dropout=args.enable_modality_dropout,
                              p_drop_eda=args.p_drop_eda, p_drop_cheap=args.p_drop_cheap,
                              val_without_eda=args.val_without_eda, val_with_dropout=args.val_with_dropout,
                              mask_ratio=args.mask_ratio)
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_total_loss:.4f} (Align: {avg_train_align_loss:.4f}, Disent: {avg_train_disent_loss:.4f}) | Val Loss: {val_losses['total']:.4f} (Align: {val_losses['align']:.4f}, Disent: {val_losses['disent']:.4f})")

        # Update learning rate
        for sch in schedulers.values():
            sch.step()

        # --- Save Losses and Checkpoints ---
        #urrent_lr = schedulers['ecg'].get_last_lr()[0] # Get LR from one of the schedulers

        any_name = next(iter(schedulers.keys()))
        current_lr = schedulers[any_name].get_last_lr()[0]  # Get LR from any scheduler

        losses.append({
            'epoch': epoch + 1,
            'train_total_loss': avg_train_total_loss, 'train_recon_loss': avg_train_recon_loss, 'train_align_loss': avg_train_align_loss, 'train_disent_loss': avg_train_disent_loss,
            'val_total_loss': val_losses['total'], 'val_recon_loss': val_losses['recon'], 'val_align_loss': val_losses['align'], 'val_disent_loss': val_losses['disent'],
            'lr': current_lr
        })
        losses_df = pd.DataFrame(losses)
        
        # Save losses as a .npz file
        loss_path = os.path.join(run_output_path, "losses.npz")
        np.savez(loss_path,
                 epoch=losses_df['epoch'].values,
                 train_total_loss=losses_df['train_total_loss'].values,
                 train_recon_loss=losses_df['train_recon_loss'].values,
                 train_align_loss=losses_df['train_align_loss'].values,
                 val_total_loss=losses_df['val_total_loss'].values,
                 val_recon_loss=losses_df['val_recon_loss'].values,
                 val_align_loss=losses_df['val_align_loss'].values,
                 lr=losses_df['lr'].values)

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(models_path, "best_ckpt.pt")
            save_checkpoint(epoch + 1, models, optimizers, schedulers, best_val_loss, losses_df.to_dict('records'), best_model_path)
        
        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_path = os.path.join(run_output_path, f"loss_curves.png")
            plot_mae_losses(losses_df, plot_path)
            logging.info(f"Saved loss plot to {plot_path}")
            
            # Plot reconstructions
            recon_plot_path = os.path.join(run_output_path, f"reconstructions_epoch_{epoch+1}.png")
            plot_reconstructions(models, vis_batch, device, recon_plot_path)
            logging.info(f"Saved reconstruction plot to {recon_plot_path}")

    logging.info("Training complete.")
    # --- Final Plot ---
    final_plot_path = os.path.join(run_output_path, "loss_curves.png")
    plot_mae_losses(pd.DataFrame(losses), final_plot_path)
    logging.info(f"Saved final loss plot to {final_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a Multi-Modal Cheap Sensor MAE')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/j-jepa-vol/PULSE/results/cheap_maes", help='Directory to save logs and models')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for training/validation')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_percentage', type=float, default=1.0, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")
    parser.add_argument('--alignment_loss_weight', type=float, default=1.0, help="Weight for the alignment loss component.")
    parser.add_argument('--disentangling_loss_weight', type=float, default=0.0, help="Weight for the disentangling loss (cosine similarity between private and shared embeddings).")
    parser.add_argument('--device', type=str, default='cuda:15' if torch.cuda.is_available() else 'cpu', help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    
    # Model specific arguments
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension of the encoder.')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the encoder.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in the encoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='Embedding dimension of the decoder.')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Depth of the decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='Number of heads in the decoder.')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio for the encoder.')
    parser.add_argument('--decoder_mlp_ratio', type=float, default=4.0, help='MLP ratio for the decoder.')
    parser.add_argument('--big_model', action='store_true', help='Use a bigger model configuration.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--lr_restart_epochs', type=int, default=20, help='Number of epochs for the first restart in CosineAnnealingWarmRestarts scheduler (T_0).')
    parser.add_argument('--t_mult', type=int, default=1, help='Factor by which to increase T_i after a restart. T_mult=1 is constant.')
    parser.add_argument('--private_mask_ratio', type=float, default=0.5, help='Ratio of private to shared embeddings')
    parser.add_argument('--hinge_alpha', type=float, default=0.2, help='Margin for the hinge loss.')
    parser.add_argument('--hinge_neg_sample_percent', type=float, default=None, help='Percentage of hard negatives for hinge loss (0.0 to 1.0). Default is None (use all).')
    parser.add_argument('--use_curriculum', action='store_true', help='Enable curriculum learning for loss weighting, gradually shifting from reconstruction to alignment.')
    parser.add_argument('--include_eda', action='store_true', help='Include EDA modality in pretraining in addition to ecg, bvp, acc, temp.')
    parser.add_argument('--modalities', type=str, default='all', help='Comma-separated list of modalities from {ecg,bvp,acc,temp,eda}, or "all"')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Patch masking ratio for MAE input.')
    # Modality dropout controls
    parser.add_argument('--enable_modality_dropout', action='store_true', help='Enable stochastic dropping of modalities per mini-batch.')
    parser.add_argument('--p_drop_eda', type=float, default=0.0, help='Probability to drop EDA per mini-batch (if present).')
    parser.add_argument('--p_drop_cheap', type=float, default=0.0, help='Probability to drop each cheap modality per mini-batch.')
    parser.add_argument('--val_without_eda', action='store_true', help='During validation, exclude EDA from inputs.')
    parser.add_argument('--val_with_dropout', action='store_true', help='During validation, apply the same modality dropout as training.')
    args = parser.parse_args()

    setup_logging(args.run_name, args.output_path)
    train(args)

if __name__ == '__main__':
    main() 

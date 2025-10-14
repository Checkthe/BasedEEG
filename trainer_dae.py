import os
import time
import argparse
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modeling.DAE import DenoisingAutoencoder, DAELoss
from utils.loading_feature import load_dae
from config import cfg

def seed_everything(seed: int = 42) -> None:

    import os as _os
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _os.environ['PYTHONHASHSEED'] = str(seed)

    # Deterministic CuDNN settings (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    Collate function converts a list of arrays (C, N) to tensor (B, C, N).
    Keeps function name unchanged.
    """
    batch_t = [torch.as_tensor(b, dtype=torch.float32) for b in batch]
    return torch.stack(batch_t, dim=0)


def _save_checkpoint(path: str, ckpt: dict) -> None:
    """Small helper to save checkpoint atomically."""
    tmp = f"{path}.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# ----------------------
# Validation
# ----------------------
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate reconstruction MSE and feature MSE on dataloader.
    The model is expected to support:
      - forward(x, return_features=True, corrupt_input=True) -> dict with 'recon' and 'features' and 'noisy'
      - encode_features(x) -> clean features
    Returns:
        (avg_recon_mse, avg_feat_mse)
    """
    model.eval()
    mse = nn.MSELoss(reduction='mean')

    total_recon = 0.0
    total_feat_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)  # (B, C, N)
            out = model(x, return_features=True, corrupt_input=True)  # simulate corruption
            recon = out['recon']
            feat_noisy = out['features']
            feat_clean = model.encode_features(x)

            bsz = x.size(0)
            total_recon += mse(recon, x).item() * bsz
            total_feat_mse += mse(feat_noisy, feat_clean).item() * bsz
            total_samples += bsz

    if total_samples == 0:
        return float('nan'), float('nan')

    return total_recon / total_samples, total_feat_mse / total_samples

# ----------------------Training----------------------
def train(args) -> None:
    """
    Main training loop. Preserves original variable and function names.
    """
    # device selection (explicit and respects --no-cuda flag)
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    seed_everything(args.seed)

    # Ensure dataset is available (train_ds expected to be defined in main scope)
    try:
        train_ds  # noqa: F401
    except NameError:
        raise RuntimeError("train_ds is not defined. Make sure load_dae() is called before parse_args()/train().")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )

    # Model / Loss / Optimizer / Scheduler
    model = DenoisingAutoencoder(
        in_channels=args.channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        noise_std=args.noise_std,
        use_time_mask=args.use_time_mask,
        mask_ratio=args.mask_ratio
    ).to(device)

    criterion = DAELoss(
        model,
        alpha_recon=args.alpha_recon,
        alpha_feat=args.alpha_feat,
        alpha_sparsity=args.alpha_sparsity,
        alpha_contractive=0.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Logging & checkpoint directories
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"dae_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_recon = float("inf")
    start_epoch = 1

    # Resume from checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get('model_state', {}))
        optimizer.load_state_dict(ckpt.get('optim_state', {}))
        start_epoch = ckpt.get("epoch", start_epoch - 1) + 1
        best_val_recon = ckpt.get("best_val_recon", best_val_recon)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_feat = 0.0
        num_samples = 0

        epoch_start = time.time()
        for i, batch in enumerate(train_loader, 1):
            x = batch.to(device)  # (B, C, N)
            optimizer.zero_grad()

            # Forward (model internally corrupts when training)
            out = model(x, return_features=True, corrupt_input=True)
            recon = out['recon']
            feat_noisy = out['features']
            feat_clean = model.encode_features(x)

            loss = criterion(clean=x, noisy=out.get('noisy', x), recon=recon,
                             feat_noisy=feat_noisy, feat_clean=feat_clean)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bsz = x.size(0)
            running_loss += loss.item() * bsz
            # For reporting use mean MSE per batch
            recon_mse = nn.functional.mse_loss(recon, x, reduction='mean').item()
            feat_mse = nn.functional.mse_loss(feat_noisy, feat_clean, reduction='mean').item()
            running_recon += recon_mse * bsz
            running_feat += feat_mse * bsz
            num_samples += bsz

            if i % args.log_interval == 0:
                avg_loss = running_loss / num_samples
                avg_recon = running_recon / num_samples
                avg_feat = running_feat / num_samples
                print(f"Epoch[{epoch}/{args.epochs}] Step[{i}/{len(train_loader)}] "
                      f"loss={avg_loss:.6f} recon_mse={avg_recon:.6f} feat_mse={avg_feat:.6f}")

        epoch_time = time.time() - epoch_start

        # compute epoch averages
        train_loss = running_loss / num_samples if num_samples > 0 else float('nan')
        train_recon = running_recon / num_samples if num_samples > 0 else float('nan')
        train_feat = running_feat / num_samples if num_samples > 0 else float('nan')

        # TensorBoard logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/recon_mse', train_recon, epoch)
        writer.add_scalar('train/feat_mse', train_feat, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch} done in {epoch_time:.1f}s. Train loss {train_loss:.6f} recon {train_recon:.6f}.")

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_recon': best_val_recon
        }
        try:
            _save_checkpoint(cfgs.save_dae, ckpt)
        except Exception:
            torch.save(ckpt, cfgs.save_dae)

    writer.close()
    print("Training finished. Best val recon:", best_val_recon)


# ----------------------
# Argument parser
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Denoising Autoencoder")
    p.add_argument("--dataset", type=str, default="synthetic", help="dataset (synthetic or custom)")
    p.add_argument("--train-samples", type=int, default=1024)
    p.add_argument("--val-samples", type=int, default=256)
    # NOTE: default for --channels set at runtime in __main__ (matches original behavior)
    p.add_argument("--channels", type=int, default=channels)
    p.add_argument("--latent-dim", type=int, default=1280)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--use-time-mask", type=bool, default=True)
    p.add_argument("--mask-ratio", type=float, default=0.12)
    p.add_argument("--alpha-recon", type=float, default=1.0)
    p.add_argument("--alpha-feat", type=float, default=0.5)
    p.add_argument("--alpha-sparsity", type=float, default=1e-4)
    p.add_argument("--batch-norm-affine", type=bool, default=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-dir", type=str, default="./logs")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    p.add_argument("--no-cuda", action="store_true", help="disable cuda even if available")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--early-stop", type=int, default=0, help="epochs of no improvement to early stop (0=disable)")
    return p.parse_args()


if __name__ == "__main__":
    cfgs = cfg().get_args()
    train_ds = load_dae()
    channels = train_ds.data.shape[1]
    args = parse_args()
    train(args)

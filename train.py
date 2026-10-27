"""
train.py — Backend: Training loop with checkpoint saving and loss logging
Usage (CLI):
    python train.py --epochs 10 --batch_size 64 --subset_size 10000
"""

import argparse
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import Generator, Discriminator, weights_init, LATENT_DIM
from preprocessing import get_dataloader

CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR     = "samples"
LOG_FILE       = "loss_log.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR,     exist_ok=True)


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-4,
    subset_size: int = 10000,
    save_every: int = 2,
    progress_callback=None,   # callable(epoch, epochs, g_loss, d_loss) for Streamlit
) -> list[dict]:
    """
    Train the DCGAN. Returns loss history list.
    progress_callback is optional — called after each epoch for live UI updates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  epochs={epochs}  batch={batch_size}  subset={subset_size}")

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # ── Loss & optimisers ─────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Fixed noise to track visual progress epoch-by-epoch
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = get_dataloader(batch_size, subset_size)
    loss_history = []

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        g_losses, d_losses = [], []

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # Train Discriminator
            D.zero_grad()
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            loss_d_real = criterion(D(real_imgs), real_labels)
            loss_d_real.backward()

            noise = torch.randn(bsz, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(noise)
            loss_d_fake = criterion(D(fake_imgs.detach()), fake_labels)
            loss_d_fake.backward()
            loss_D = loss_d_real + loss_d_fake
            opt_D.step()

            # Train Generator
            G.zero_grad()
            loss_G = criterion(D(fake_imgs), real_labels)  # G wants D to say real
            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "g_loss": round(avg_g, 4), "d_loss": round(avg_d, 4),
               "time_s": round(elapsed, 1)}
        loss_history.append(row)

        print(f"Epoch [{epoch:>3}/{epochs}]  G={avg_g:.4f}  D={avg_d:.4f}  {elapsed:.1f}s")

        # Save sample images and checkpoints
        if epoch % save_every == 0 or epoch == epochs:
            with torch.no_grad():
                samples = G(fixed_noise)
            save_image(samples, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png",
                       nrow=8, normalize=True)
            torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_epoch{epoch}.pth")
            torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_epoch{epoch}.pth")

        # Streamlit live callback
        if progress_callback:
            progress_callback(epoch, epochs, avg_g, avg_d)

    # Save loss log
    with open(LOG_FILE, "w") as f:
        json.dump(loss_history, f, indent=2)

    print(f"\nTraining complete. Checkpoints → {CHECKPOINT_DIR}/  Samples → {SAMPLE_DIR}/")
    return loss_history


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--subset_size", type=int,   default=10000)
    p.add_argument("--save_every",  type=int,   default=2)
    args = p.parse_args()
    train(**vars(args))

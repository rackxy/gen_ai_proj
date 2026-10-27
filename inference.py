"""
inference.py — Backend: Model loading and inference
- load_generator()   : load trained Generator checkpoint
- generate_images()  : run Generator inference
- score_image()      : run Discriminator to score real/fake probability
"""

import os
import glob
import torch
import numpy as np

from model import Generator, Discriminator, LATENT_DIM, weights_init
from preprocessing import tensor_to_display, make_grid_array


CHECKPOINT_DIR = "checkpoints"


# ── Model loading ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(checkpoint_path: str) -> Generator:
    """Load a trained Generator from a .pth checkpoint file."""
    device = get_device()
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def load_discriminator(checkpoint_path: str) -> Discriminator:
    """Load a trained Discriminator from a .pth checkpoint file."""
    device = get_device()
    D = Discriminator().to(device)
    D.load_state_dict(torch.load(checkpoint_path, map_location=device))
    D.eval()
    return D


def list_generator_checkpoints() -> list[str]:
    """Return sorted list of Generator checkpoint paths."""
    return sorted(glob.glob(f"{CHECKPOINT_DIR}/G_epoch*.pth"))


def list_discriminator_checkpoints() -> list[str]:
    """Return sorted list of Discriminator checkpoint paths."""
    return sorted(glob.glob(f"{CHECKPOINT_DIR}/D_epoch*.pth"))


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_images(
    generator: Generator,
    n: int = 16,
    seed: int | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Generate n fashion images using the loaded Generator.

    Args:
        generator   : loaded Generator model
        n           : number of images to generate
        seed        : random seed for reproducibility (None = random)
        temperature : scale the noise std (>1 = more diverse, <1 = more uniform)

    Returns:
        tensors : (N, 1, 28, 28) torch.Tensor in [-1,1]
        grid    : HxW numpy uint8 grid image for display
    """
    device = get_device()
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn(n, LATENT_DIM, 1, 1, device=device) * temperature
    with torch.no_grad():
        fake = generator(noise)

    grid = make_grid_array(fake, nrow=8)
    return fake, grid


def score_image(discriminator: Discriminator, tensor: torch.Tensor) -> float:
    """
    Score an image tensor with the Discriminator.
    Returns a float in [0, 1]:  1.0 = looks real, 0.0 = looks fake.
    """
    device = get_device()
    tensor = tensor.to(device)
    with torch.no_grad():
        score = discriminator(tensor).item()
    return score


def diversity_score(tensors: torch.Tensor) -> float:
    """
    Mean pairwise L2 distance across generated images — higher = more diverse.
    """
    flat = tensors.view(len(tensors), -1).cpu().numpy()
    dists = []
    for i in range(min(len(flat), 20)):
        for j in range(i + 1, min(len(flat), 20)):
            dists.append(float(np.linalg.norm(flat[i] - flat[j])))
    return float(np.mean(dists)) if dists else 0.0

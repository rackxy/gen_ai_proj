"""
preprocessing.py — Backend: Data loading and image preprocessing
"""

import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Normalise to [-1, 1] to match Generator Tanh output
TRANSFORM = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# For uploaded images: resize + grayscale + normalise
UPLOAD_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def get_dataloader(batch_size: int = 64, subset_size: int = 10000) -> DataLoader:
    """Download Fashion-MNIST and return a DataLoader (subset for speed)."""
    dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=TRANSFORM
    )
    if subset_size and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=True)


def preprocess_uploaded_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a user-uploaded PIL image to a model-ready tensor.
    Returns shape (1, 1, 28, 28) normalised to [-1, 1].
    """
    tensor = UPLOAD_TRANSFORM(pil_img)
    return tensor.unsqueeze(0)  # add batch dim


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW tensor in [-1,1] to a HW uint8 numpy array for display.
    """
    arr = tensor.squeeze().cpu().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return arr


def make_grid_array(tensors: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """
    Lay out a batch of image tensors into a grid as a numpy array.
    tensors: (N, 1, H, W) in [-1,1]
    """
    imgs = [(((t.squeeze().numpy() + 1) / 2) * 255).clip(0, 255).astype(np.uint8)
            for t in tensors.cpu()]
    n    = len(imgs)
    ncol = min(nrow, n)
    nrow_ = (n + ncol - 1) // ncol
    H, W = imgs[0].shape
    grid = np.ones((nrow_ * H, ncol * W), dtype=np.uint8) * 255
    for i, img in enumerate(imgs):
        r, c = divmod(i, ncol)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
    return grid

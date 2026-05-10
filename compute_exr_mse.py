from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyexr


def load_exr(path: Path) -> np.ndarray:
    image = pyexr.read(str(path))
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[..., None]
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE between an EXR output image and a reference image while ignoring invalid pixels."
    )
    parser.add_argument("image", type=Path, help="Path to the EXR image to evaluate.")
    parser.add_argument("reference", type=Path, help="Path to the reference EXR image.")
    args = parser.parse_args()

    image = load_exr(args.image)
    reference = load_exr(args.reference)

    if image.shape != reference.shape:
        raise ValueError(f"Image shape mismatch: {image.shape} vs {reference.shape}")

    valid_mask = np.all(np.isfinite(image) & np.isfinite(reference), axis=-1)
    valid_pixel_count = int(np.count_nonzero(valid_mask))
    total_pixel_count = int(valid_mask.size)

    if valid_pixel_count == 0:
        raise ValueError("No valid pixels remain after filtering NaN/Inf values.")

    diff = image[valid_mask] - reference[valid_mask]
    mse = float(np.mean(diff * diff))

    print(f"image: {args.image}")
    print(f"reference: {args.reference}")
    print(f"resolution: {image.shape[1]}x{image.shape[0]}")
    print(f"channels: {image.shape[2]}")
    print(f"valid pixels: {valid_pixel_count} / {total_pixel_count}")
    print(f"ignored pixels: {total_pixel_count - valid_pixel_count}")
    print(f"MSE: {mse:.12g}")


if __name__ == "__main__":
    main()

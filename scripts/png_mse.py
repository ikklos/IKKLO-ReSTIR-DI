import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_png(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    return array


def save_error_png(path: Path, error_image: np.ndarray) -> None:
    clipped = np.clip(error_image, 0.0, 255.0).astype(np.uint8)
    if clipped.shape[2] == 1:
        clipped = clipped[..., 0]
    Image.fromarray(clipped).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute PNG MSE with optional outlier filtering for fast visual-paper debugging."
    )
    parser.add_argument("image", type=Path, help="PNG image to evaluate")
    parser.add_argument("reference", type=Path, help="Reference PNG image")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path for the per-channel absolute error image",
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=None,
        help="Ignore pixels whose maximum per-channel absolute error is larger than this threshold",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Ignore pixels if any channel in either input image is larger than this threshold",
    )
    args = parser.parse_args()

    image = load_png(args.image)
    reference = load_png(args.reference)

    if image.shape != reference.shape:
        raise ValueError(f"Image shape mismatch: {image.shape} vs {reference.shape}")

    finite_mask = np.all(np.isfinite(image) & np.isfinite(reference), axis=-1)
    value_mask = np.ones_like(finite_mask, dtype=bool)
    if args.max_value is not None:
        value_mask = np.all((image <= args.max_value) & (reference <= args.max_value), axis=-1)

    abs_error = np.abs(image - reference)
    error_mask = np.ones_like(finite_mask, dtype=bool)
    if args.max_abs_error is not None:
        error_mask = np.max(abs_error, axis=-1) <= args.max_abs_error

    valid_mask = finite_mask & value_mask & error_mask
    valid_pixel_count = int(np.count_nonzero(valid_mask))
    total_pixel_count = int(valid_mask.size)

    if valid_pixel_count == 0:
        raise ValueError("No valid pixels remain after filtering.")

    mse = float(np.mean(np.square(image[valid_mask] - reference[valid_mask], dtype=np.float64)))

    if args.output is not None:
        error_vis = np.zeros_like(abs_error, dtype=np.float32)
        error_vis[valid_mask] = abs_error[valid_mask]
        save_error_png(args.output, error_vis)

    print(f"image: {args.image}")
    print(f"reference: {args.reference}")
    print(f"resolution: {image.shape[1]}x{image.shape[0]}")
    print(f"channels: {image.shape[2]}")
    print(f"valid pixels: {valid_pixel_count} / {total_pixel_count}")
    print(f"ignored pixels: {total_pixel_count - valid_pixel_count}")
    if args.max_abs_error is not None:
        print(f"max abs error threshold: {args.max_abs_error}")
    if args.max_value is not None:
        print(f"max value threshold: {args.max_value}")
    if args.output is not None:
        print(f"error image: {args.output}")
    print(f"MSE: {mse:.12g}")


if __name__ == "__main__":
    main()

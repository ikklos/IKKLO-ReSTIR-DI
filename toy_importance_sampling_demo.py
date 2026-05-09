from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_mix(x: np.ndarray, centers: list[float], widths: list[float], amps: list[float]) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    for c, w, a in zip(centers, widths, amps):
        y += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    return y


def normalize_pdf(x: np.ndarray, raw_pdf: np.ndarray) -> np.ndarray:
    clipped = np.maximum(raw_pdf, 1e-12)
    integral = np.trapezoid(clipped, x)
    return clipped / integral


def build_problem(x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    # Sampling distribution p(x): smooth, multi-peak, and strongly aligned with f1/f2.
    raw_p = (
        0.08
        + gaussian_mix(x, [0.16, 0.42, 0.71], [0.055, 0.095, 0.075], [1.25, 1.0, 1.15])
        + 0.06 * gaussian_mix(x, [0.88], [0.14], [1.0])
    )
    p = normalize_pdf(x, raw_p)

    # Three smooth integrands on the same domain [0, 1].
    # f1/f2 are correlated with p(x), while f3 deliberately places energy away from p(x)'s main peaks.
    f1 = (
        0.16
        + 0.85 * gaussian_mix(x, [0.15, 0.44, 0.70], [0.06, 0.085, 0.08], [1.0, 0.75, 1.1])
        + 0.045 * gaussian_mix(x, [0.28, 0.58], [0.03, 0.04], [1.0, 0.8])
        + 0.018 * np.sin(2.0 * np.pi * (1.4 * x + 0.1)) ** 2
    )

    f2 = (
        0.12
        + 0.75 * gaussian_mix(x, [0.18, 0.40, 0.66], [0.07, 0.11, 0.09], [0.8, 1.0, 0.75])
        + 0.20 * gaussian_mix(x, [0.79], [0.045], [1.0])
        + 0.020 * np.cos(2.0 * np.pi * (1.1 * x - 0.25)) ** 2
    )

    f3 = (
        0.14
        + 0.95 * gaussian_mix(x, [0.06, 0.87], [0.035, 0.05], [1.0, 0.9])
        + 0.24 * gaussian_mix(x, [0.95], [0.02], [1.0])
        + 0.016 * np.sin(2.0 * np.pi * (2.2 * x + 0.4)) ** 2
    )

    return p, [f1, f2, f3], ["f1(x)", "f2(x)", "f3(x)"]


def sample_from_tabulated_pdf(x: np.ndarray, pdf: np.ndarray, sample_count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cdf = np.empty_like(pdf)
    cdf[0] = 0.0
    dx = np.diff(x)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
    cdf[-1] = 1.0

    u = rng.random(sample_count)
    samples = np.interp(u, cdf, x)
    sample_pdf = np.interp(samples, x, pdf)
    return samples, sample_pdf


def cumulative_importance_estimate(values: np.ndarray) -> np.ndarray:
    partial_sum = np.cumsum(values, dtype=np.float64)
    counts = np.arange(1, values.size + 1, dtype=np.float64)
    return partial_sum / counts


def save_line_plot(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_error_plot(
    iterations: np.ndarray,
    error: np.ndarray,
    title: str,
    path: Path,
    color: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot(iterations, error, color=color, linewidth=1.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy importance sampling demo with one sampling distribution and three smooth integrands.")
    parser.add_argument("--samples", type=int, default=50000, help="Number of Monte Carlo samples.")
    parser.add_argument("--grid", type=int, default=20000, help="Number of grid points for tabulation and reference integration.")
    parser.add_argument("--repeats", type=int, default=64, help="Number of independent experiments used to estimate RMSE.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "toy_is_output",
        help="Directory used to save figures.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, args.grid, dtype=np.float64)
    p, functions, names = build_problem(x)

    rng = np.random.default_rng(args.seed)
    samples, sample_pdf = sample_from_tabulated_pdf(x, p, args.samples, rng)

    save_line_plot(
        x,
        p,
        title="Sampling Distribution p(x)",
        xlabel="x",
        ylabel="p(x)",
        path=output_dir / "01_px.png",
        color="#1f77b4",
    )

    function_colors = ["#d62728", "#2ca02c", "#9467bd"]
    error_colors = ["#ff7f0e", "#17becf", "#8c564b"]

    print(f"Output directory: {output_dir}")
    print(f"Sample count: {args.samples}")
    print(f"Repeat count: {args.repeats}")
    print("Importance sampling results")
    print("-" * 72)
    print(f"{'Function':<10} {'Truth':>14} {'Mean Est':>14} {'Final RMSE':>14} {'Rel RMSE':>14}")

    results: list[dict[str, object]] = []
    for idx, (f, name, f_color, e_color) in enumerate(zip(functions, names, function_colors, error_colors), start=1):
        truth = np.trapezoid(f, x)
        running_estimates = np.empty((args.repeats, args.samples), dtype=np.float64)
        for repeat in range(args.repeats):
            repeat_samples, repeat_pdf = sample_from_tabulated_pdf(x, p, args.samples, rng)
            fx_at_samples = np.interp(repeat_samples, x, f)
            is_values = fx_at_samples / np.maximum(repeat_pdf, 1e-12)
            running_estimates[repeat] = cumulative_importance_estimate(is_values)

        rmse_curve = np.sqrt(np.mean((running_estimates - truth) ** 2, axis=0))
        mean_final_estimate = float(np.mean(running_estimates[:, -1]))
        final_rmse = float(rmse_curve[-1])
        relative_rmse = final_rmse / max(abs(truth), 1e-12)

        print(f"{name:<10} {truth:>14.8f} {mean_final_estimate:>14.8f} {final_rmse:>14.8f} {relative_rmse:>14.8f}")

        save_line_plot(
            x,
            f,
            title=f"{name} on [0, 1]",
            xlabel="x",
            ylabel=name,
            path=output_dir / f"{idx + 1:02d}_{name.replace('(x)', '').lower()}.png",
            color=f_color,
        )

        results.append(
            {
                "idx": idx,
                "name": name,
                "rmse_curve": rmse_curve,
                "color": e_color,
            }
        )

    global_rmse_min = min(float(np.min(item["rmse_curve"])) for item in results)
    global_rmse_max = max(float(np.max(item["rmse_curve"])) for item in results)
    y_limits = (
        max(global_rmse_min * 0.9, 1e-16),
        max(global_rmse_max * 1.1, global_rmse_min * 1.5),
    )

    for item in results:
        save_error_plot(
            np.arange(1, args.samples + 1),
            item["rmse_curve"] + 1e-16,
            title=f"{item['name']}: Iteration vs RMSE",
            path=output_dir / f"{int(item['idx']) + 4:02d}_{str(item['name']).replace('(x)', '').lower()}_rmse.png",
            color=str(item["color"]),
            y_limits=y_limits,
        )


if __name__ == "__main__":
    main()

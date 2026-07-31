from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_aas_matrices(
    epoch_data: np.ndarray,
    averaging_matrix: np.ndarray,
    sfreq: float | None,
    threshold: float,
    target_epoch: int = 0,
    output_path: str | Path | None = None,
    show_plot: bool = True,
    *,
    epoch_index: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Visualize AAS template diagnostics for direct implementation comparison."""
    if epoch_index is not None:
        target_epoch = epoch_index
    if save_path is not None and output_path is None:
        output_path = save_path

    D = np.asarray(epoch_data, dtype=float)
    A = np.asarray(averaging_matrix, dtype=float)

    if D.ndim != 2:
        raise ValueError("epoch_data must be a 2-D matrix.")
    if A.shape != (D.shape[0], D.shape[0]):
        raise ValueError(
            "averaging_matrix must have shape "
            f"({D.shape[0]}, {D.shape[0]}), got {A.shape}."
        )
    if not 0 <= target_epoch < D.shape[0]:
        raise IndexError("target_epoch is outside the available epoch range.")

    threshold = float(threshold)
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie between -1 and 1.")

    D_uV = D * 1e6
    N = A @ D_uV
    R = D_uV - N

    time = np.arange(D.shape[1]) / sfreq if sfreq is not None else np.arange(D.shape[1])
    time_label = "Time within epoch (s)" if sfreq is not None else "Sample"

    amplitude_limit = np.nanpercentile(np.abs(D_uV), 99)
    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        amplitude_limit = max(float(np.nanmax(np.abs(D_uV))), 1.0)
    residual_limit = np.nanpercentile(np.abs(R), 99)
    if not np.isfinite(residual_limit) or residual_limit <= 0:
        residual_limit = max(float(np.nanmax(np.abs(R))), 1.0)

    if D.shape[0] > 1:
        corr_matrix = np.corrcoef(D_uV, rowvar=True)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix[np.diag_indices_from(corr_matrix)] = 1.0
        corr_values = corr_matrix[np.triu_indices(D.shape[0], k=1)]
    else:
        corr_matrix = np.ones((1, 1), dtype=float)
        corr_values = np.array([], dtype=float)

    corr_limit = np.nanpercentile(np.abs(corr_values), 99) if corr_values.size else 1.0
    corr_limit = max(float(corr_limit), 1.0)

    selected_counts = np.count_nonzero(A > 0, axis=1)
    mean_selected_count = float(np.mean(selected_counts)) if selected_counts.size else 0.0
    fraction_above_threshold = float(np.mean(corr_values > threshold)) if corr_values.size else 0.0
    mean_self_weight = float(np.mean(np.diag(A))) if A.size else 0.0
    residual_artifact_fraction = (
        float(np.linalg.norm(R.ravel()) / np.linalg.norm(D_uV.ravel()))
        if np.linalg.norm(D_uV.ravel()) > 0
        else 0.0
    )
    artifact_reduction_db = (
        float(10.0 * np.log10((np.sum(D_uV**2) + 1e-30) / (np.sum(R**2) + 1e-30)))
        if np.sum(R**2) > 0
        else float("inf")
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("AAS template diagnostics (EEG shown in µV)", fontsize=14)

    image_d = axes[0, 0].imshow(
        D_uV,
        aspect="auto",
        origin="lower",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[0, 0].set_title("D: measured artifact epochs")
    axes[0, 0].set_xlabel(time_label)
    axes[0, 0].set_ylabel("Epoch")
    fig.colorbar(image_d, ax=axes[0, 0], label="Amplitude (µV)")

    image_a = axes[0, 1].imshow(
        A,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=max(float(np.nanmax(A)), 1.0),
    )
    axes[0, 1].set_title("A: averaging matrix")
    axes[0, 1].set_xlabel("Source epoch")
    axes[0, 1].set_ylabel("Target epoch")
    fig.colorbar(image_a, ax=axes[0, 1], label="Weight")

    image_n = axes[0, 2].imshow(
        N,
        aspect="auto",
        origin="lower",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[0, 2].set_title("N = A @ D: estimated artifact templates")
    axes[0, 2].set_xlabel(time_label)
    axes[0, 2].set_ylabel("Epoch")
    fig.colorbar(image_n, ax=axes[0, 2], label="Amplitude (µV)")

    image_r = axes[1, 0].imshow(
        R,
        aspect="auto",
        origin="lower",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[1, 0].set_title("R = D - N: residual epochs")
    axes[1, 0].set_xlabel(time_label)
    axes[1, 0].set_ylabel("Epoch")
    fig.colorbar(image_r, ax=axes[1, 0], label="Residual amplitude (µV)")

    image_c = axes[1, 1].imshow(
        corr_matrix,
        aspect="auto",
        origin="lower",
        vmin=-corr_limit,
        vmax=corr_limit,
        cmap="viridis",
    )
    axes[1, 1].set_title("Pairwise epoch correlation matrix C")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Epoch")
    fig.colorbar(image_c, ax=axes[1, 1], label="Correlation")

    bins = min(20, max(10, D.shape[0] * 2))
    axes[1, 2].hist(corr_values, bins=bins, color="steelblue", alpha=0.8)
    axes[1, 2].axvline(
        threshold,
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"thr={threshold:.2f}",
    )
    axes[1, 2].set_title("Correlation histogram")
    axes[1, 2].set_xlabel("Correlation")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].legend(loc="best")

    axes[2, 0].plot(time, D_uV[target_epoch], label="Measured epoch D")
    axes[2, 0].plot(time, N[target_epoch], label="Template N")
    axes[2, 0].plot(time, R[target_epoch], label="Residual D - N")
    axes[2, 0].set_title(f"Target epoch {target_epoch}")
    axes[2, 0].set_xlabel(time_label)
    axes[2, 0].set_ylabel("Amplitude (µV)")
    axes[2, 0].legend(loc="best")

    axes[2, 1].bar(np.arange(len(selected_counts)), selected_counts, color="seagreen")
    axes[2, 1].set_title("Selected epoch count per row of A")
    axes[2, 1].set_xlabel("Row")
    axes[2, 1].set_ylabel("Selected epochs")

    metrics_text = (
        "Compact metrics\n"
        f"mean selected count: {mean_selected_count:.3f}\n"
        f"fraction above threshold: {fraction_above_threshold:.3f}\n"
        f"mean self-weight: {mean_self_weight:.3f}\n"
        f"residual artifact fraction: {residual_artifact_fraction:.3f}\n"
        f"artifact reduction (dB): {artifact_reduction_db:.3f}"
    )
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.98,
        metrics_text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "0.95", "alpha": 0.95},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

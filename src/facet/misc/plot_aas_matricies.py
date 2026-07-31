from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_aas_matrices(
    epoch_data: np.ndarray,
    averaging_matrix: np.ndarray,
    sfreq: float | None,
    threshold: float,
    target_epoch: int = 0,
    channel_name: str | None = None,
    output_path: str | Path | None = None,
    show_plot: bool = True,
    *,
    epoch_index: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Visualize AAS template construction and correction diagnostics."""
    if epoch_index is not None:
        target_epoch = epoch_index

    if save_path is not None and output_path is None:
        output_path = save_path

    D = np.asarray(epoch_data, dtype=float)
    A = np.asarray(averaging_matrix, dtype=float)

    if D.ndim != 2:
        raise ValueError(
            f"epoch_data must be 2-D, got shape {D.shape}."
        )

    if D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError("epoch_data must not be empty.")

    if A.shape != (D.shape[0], D.shape[0]):
        raise ValueError(
            "averaging_matrix must have shape "
            f"({D.shape[0]}, {D.shape[0]}), got {A.shape}."
        )

    if not np.all(np.isfinite(D)):
        raise ValueError("epoch_data contains NaN or infinite values.")

    if not np.all(np.isfinite(A)):
        raise ValueError(
            "averaging_matrix contains NaN or infinite values."
        )

    if not 0 <= target_epoch < D.shape[0]:
        raise IndexError(
            f"target_epoch {target_epoch} is outside the range "
            f"0 to {D.shape[0] - 1}."
        )

    threshold = float(threshold)

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie between -1 and 1.")

    data_std = float(np.nanstd(D))

    if not np.isfinite(data_std) or data_std <= 1e-15:
        raise ValueError(
            "Epoch matrix D is constant or effectively zero. "
            "Select another channel or inspect epoch extraction."
        )

    if sfreq is not None and sfreq <= 0:
        raise ValueError("sfreq must be positive when supplied.")

    # Convert volts to microvolts for display.
    D_uV = D * 1e6
    N_uV = A @ D_uV
    R_uV = D_uV - N_uV

    n_epochs, n_samples = D.shape

    if sfreq is not None:
        duration = n_samples / float(sfreq)
        time = np.arange(n_samples, dtype=float) / float(sfreq)
        time_label = "Time within epoch (s)"
        heatmap_extent = [0.0, duration, 0.0, float(n_epochs)]
    else:
        time = np.arange(n_samples, dtype=float)
        time_label = "Sample within epoch"
        heatmap_extent = [
            0.0,
            float(n_samples),
            0.0,
            float(n_epochs),
        ]

    # Use a common robust scale for D and N.
    combined_signal = np.concatenate(
        [D_uV.ravel(), N_uV.ravel()]
    )
    amplitude_limit = float(
        np.nanpercentile(np.abs(combined_signal), 99)
    )

    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        amplitude_limit = float(
            np.nanmax(np.abs(combined_signal))
        )

    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        raise ValueError(
            "Measured and template amplitudes are effectively zero."
        )

    residual_limit = float(
        np.nanpercentile(np.abs(R_uV), 99)
    )

    if not np.isfinite(residual_limit) or residual_limit <= 0:
        residual_limit = float(np.nanmax(np.abs(R_uV)))

    if not np.isfinite(residual_limit) or residual_limit <= 0:
        residual_limit = amplitude_limit

    # Pairwise correlations, handling constant individual epochs safely.
    corr_matrix = np.eye(n_epochs, dtype=float)

    for i in range(n_epochs):
        x = D_uV[i]
        x_std = float(np.std(x))

        for j in range(i + 1, n_epochs):
            y = D_uV[j]
            y_std = float(np.std(y))

            if x_std <= 1e-15 or y_std <= 1e-15:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    upper_indices = np.triu_indices(n_epochs, k=1)
    corr_values = corr_matrix[upper_indices]
    corr_values = corr_values[np.isfinite(corr_values)]

    selected_counts = np.count_nonzero(
        np.abs(A) > np.finfo(float).eps,
        axis=1,
    )

    mean_selected_count = float(np.mean(selected_counts))
    fraction_above_threshold = (
        float(np.mean(corr_values > threshold))
        if corr_values.size
        else float("nan")
    )

    diagonal_weights = np.diag(A)
    mean_self_weight = float(np.mean(diagonal_weights))

    input_power = float(np.sum(D_uV**2))
    residual_power = float(np.sum(R_uV**2))

    if input_power <= np.finfo(float).eps:
        residual_artifact_fraction = float("nan")
        artifact_reduction_db = float("nan")
    else:
        residual_artifact_fraction = float(
            np.sqrt(residual_power / input_power)
        )
        artifact_reduction_db = float(
            10.0
            * np.log10(
                input_power
                / max(
                    residual_power,
                    np.finfo(float).eps,
                )
            )
        )

    row_sums = np.sum(A, axis=1)
    row_sum_errors = int(
        np.count_nonzero(
            ~np.isclose(
                row_sums,
                1.0,
                rtol=1e-6,
                atol=1e-8,
            )
        )
    )

    figure_title = "AAS template diagnostics"
    if channel_name:
        figure_title += f" — {channel_name}"
    figure_title += " (EEG shown in µV)"

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14),
    )
    fig.suptitle(figure_title, fontsize=14)

    image_d = axes[0, 0].imshow(
        D_uV,
        aspect="auto",
        origin="lower",
        extent=heatmap_extent,
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[0, 0].set_title("D: measured artifact epochs")
    axes[0, 0].set_xlabel(time_label)
    axes[0, 0].set_ylabel("Epoch")
    fig.colorbar(
        image_d,
        ax=axes[0, 0],
        label="Amplitude (µV)",
    )

    matrix_max = float(np.nanmax(np.abs(A)))
    if matrix_max <= 0:
        matrix_max = 1.0

    image_a = axes[0, 1].imshow(
        A,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=matrix_max,
    )
    axes[0, 1].set_title("A: averaging matrix")
    axes[0, 1].set_xlabel("Source epoch")
    axes[0, 1].set_ylabel("Target epoch")
    fig.colorbar(
        image_a,
        ax=axes[0, 1],
        label="Weight",
    )

    image_n = axes[0, 2].imshow(
        N_uV,
        aspect="auto",
        origin="lower",
        extent=heatmap_extent,
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[0, 2].set_title(
        "N = A @ D: estimated artifact templates"
    )
    axes[0, 2].set_xlabel(time_label)
    axes[0, 2].set_ylabel("Epoch")
    fig.colorbar(
        image_n,
        ax=axes[0, 2],
        label="Amplitude (µV)",
    )

    image_r = axes[1, 0].imshow(
        R_uV,
        aspect="auto",
        origin="lower",
        extent=heatmap_extent,
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[1, 0].set_title("R = D - N: residual epochs")
    axes[1, 0].set_xlabel(time_label)
    axes[1, 0].set_ylabel("Epoch")
    fig.colorbar(
        image_r,
        ax=axes[1, 0],
        label="Residual amplitude (µV)",
    )

    image_c = axes[1, 1].imshow(
        corr_matrix,
        aspect="auto",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[1, 1].set_title(
        "Pairwise epoch correlation matrix C"
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Epoch")
    fig.colorbar(
        image_c,
        ax=axes[1, 1],
        label="Correlation",
    )

    if corr_values.size:
        bins = min(
            50,
            max(10, int(np.sqrt(corr_values.size))),
        )
        axes[1, 2].hist(
            corr_values,
            bins=bins,
            alpha=0.8,
        )
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "No valid non-diagonal correlations",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )

    axes[1, 2].axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"threshold={threshold:.3f}",
    )
    axes[1, 2].set_title("Correlation histogram")
    axes[1, 2].set_xlabel("Correlation")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].legend(loc="best")

    axes[2, 0].plot(
        time,
        D_uV[target_epoch],
        label="Measured epoch D",
    )
    axes[2, 0].plot(
        time,
        N_uV[target_epoch],
        label="Template N",
    )
    axes[2, 0].plot(
        time,
        R_uV[target_epoch],
        label="Residual D - N",
    )
    axes[2, 0].set_title(
        f"Target epoch {target_epoch}"
    )
    axes[2, 0].set_xlabel(time_label)
    axes[2, 0].set_ylabel("Amplitude (µV)")
    axes[2, 0].legend(loc="best")

    axes[2, 1].bar(
        np.arange(n_epochs),
        selected_counts,
    )
    axes[2, 1].set_title(
        "Selected epoch count per row of A"
    )
    axes[2, 1].set_xlabel("Target epoch")
    axes[2, 1].set_ylabel("Selected epochs")

    metrics_text = (
        "Compact metrics\n"
        f"channel: {channel_name or 'unknown'}\n"
        f"D range: {np.min(D_uV):.3f} to "
        f"{np.max(D_uV):.3f} µV\n"
        f"D std: {np.std(D_uV):.3f} µV\n"
        f"mean selected count: "
        f"{mean_selected_count:.3f}\n"
        f"fraction above threshold: "
        f"{fraction_above_threshold:.3f}\n"
        f"mean self-weight: "
        f"{mean_self_weight:.3f}\n"
        f"row-sum errors: "
        f"{row_sum_errors}\n"
        f"residual fraction: "
        f"{residual_artifact_fraction:.3f}\n"
        f"reduction: "
        f"{artifact_reduction_db:.3f} dB"
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
        bbox={
            "boxstyle": "round",
            "facecolor": "0.95",
            "alpha": 0.95,
        },
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    plt.close(fig)
"""Plot training and validation loss histories stored in a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .train_world_model import load_checkpoint


def plot_training_history(
    checkpoint_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Render train/validation loss curves and return the PNG path."""

    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint.train_losses:
        raise ValueError("checkpoint does not contain a training-loss history")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(checkpoint.train_losses) + 1)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, checkpoint.train_losses, label="Train total", linewidth=2)
    if checkpoint.validation_losses:
        axis.plot(
            epochs,
            checkpoint.validation_losses,
            label="Validation total",
            linewidth=2,
        )
        best_loss = checkpoint.validation_losses[checkpoint.best_epoch - 1]
        axis.scatter(
            [checkpoint.best_epoch],
            [best_loss],
            label=f"Best epoch: {checkpoint.best_epoch}",
            color="#d1495b",
            zorder=3,
        )

    has_rollout_history = any(
        loss != 0.0
        for loss in (
            checkpoint.train_rollout_losses
            + checkpoint.validation_rollout_losses
        )
    )
    if has_rollout_history:
        axis.plot(
            epochs,
            checkpoint.train_one_step_losses,
            label="Train one-step",
            linestyle="--",
        )
        axis.plot(
            epochs,
            checkpoint.train_rollout_losses,
            label="Train rollout",
            linestyle=":",
        )
        if checkpoint.validation_losses:
            axis.plot(
                epochs,
                checkpoint.validation_one_step_losses,
                label="Validation one-step",
                linestyle="--",
            )
            axis.plot(
                epochs,
                checkpoint.validation_rollout_losses,
                label="Validation rollout",
                linestyle=":",
            )

    axis.set(
        xlabel="Epoch",
        ylabel="Normalized MSE loss",
        title="World Model Training History",
        yscale="log",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/world_model.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_loss.png"),
    )
    args = parser.parse_args()

    try:
        output = plot_training_history(args.checkpoint, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"saved training history to {output}")


if __name__ == "__main__":
    main()

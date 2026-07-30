#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from CFTD.gcn import train_contrastive_model

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mp_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lattice/graph encoder on MP-20 with Pettifor augmentation."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "train.csv",
        help=f"Path to the training CSV (default: {DEFAULT_DATA_DIR / 'train.csv'}).",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "val.csv",
        help=f"Path to the validation CSV (default: {DEFAULT_DATA_DIR / 'val.csv'}).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "gcn_mp20_petti.pt",
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to store the validation curve plot.",
    )

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory for faster host-to-device transfers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Prefetch batches per worker (only when num-workers > 0).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature.")
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--num-rbf", type=int, default=128, help="Number of RBF features per edge."
    )
    parser.add_argument(
        "--cutoff", type=float, default=5.0, help="cutoff for graph building."
    )
    parser.add_argument(
        "--gamma", type=float, default=20.0, help="width of RBF bases."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="Number of EGNN layers to stack."
    )
    parser.add_argument(
        "--model-type",
        choices=("equivariant", "non_equivariant"),
        default="equivariant",
        help="Encoder architecture to train.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string passed to train_contrastive_model (default: auto).",
    )
    parser.add_argument(
        "--pettifor-nn-prob",
        type=float,
        default=0.1,
        help="Probability of nearest-neighbor Pettifor substitution in augment_similar.",
    )
    parser.add_argument(
        "--pettifor-nnn-prob",
        type=float,
        default=0.05,
        help="Probability of next-nearest Pettifor substitution in augment_similar.",
    )
    parser.add_argument(
        "--pettifor-scale-path",
        type=Path,
        default=PROJECT_ROOT / "CFTD" / "files" / "mod_petti.json",
        help="JSON path for modified Pettifor scale used by augment_similar.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.2,
        help="Probability of adding wrapped fractional jitter noise in augmentation.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.005,
        help="Std of wrapped fractional jitter noise in augmentation.",
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use Hugging Face Accelerate to manage devices/distributed training.",
    )
    return parser.parse_args()


def verify_csv(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label} CSV at {path}. Run scripts/download_mp20.py first "
            "or point --train-csv / --val-csv to existing files."
        )


def main() -> None:
    args = parse_args()
    verify_csv(args.train_csv, "train")
    verify_csv(args.val_csv, "val")

    accelerator = None
    if args.accelerate:
        try:
            from accelerate import Accelerator
        except ImportError as exc:  # pragma: no cover - optional dep
            raise SystemExit(
                "Accelerate is not installed. Install it via `pip install accelerate` "
                "or rerun without --accelerate."
            ) from exc
        accelerator = Accelerator()

    def _is_main():
        return accelerator is None or accelerator.is_main_process

    if _is_main():
        print("🚀 Launching MP-20 training (Pettifor augmentation only).")

    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        gamma=args.gamma,
        n_layers=args.n_layers,
        model_type=args.model_type,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
        plot_path=str(args.plot_path) if args.plot_path else None,
        accelerator=accelerator,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    train_contrastive_model(
        train_csv=str(args.train_csv),
        val_csv=str(args.val_csv),
        use_pettifor_augmentation=True,
        pettifor_nn_prob=args.pettifor_nn_prob,
        pettifor_nnn_prob=args.pettifor_nnn_prob,
        noise_prob=args.noise_prob,
        noise_scale=args.noise_scale,
        pettifor_scale_path=str(args.pettifor_scale_path),
        **common_kwargs,
    )

    if _is_main():
        print(f"✅ Training finished. Checkpoint saved to {args.checkpoint_path}.")


if __name__ == "__main__":
    main()

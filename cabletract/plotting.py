"""Matplotlib plotting helpers used by the legacy v1 sweeps and the new
analysis phases.

These helpers were previously inlined in ``cabletract_analysis.py:310-363``.
They are kept here so that they can be unit tested with a non-interactive
backend, and so that future phases can reuse them without re-importing the
top-level monolith.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

# Force a non-interactive backend so the package works in headless CI / tests.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_dual_line_plot(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    xlabel: str,
    y1label: str,
    y2label: str,
    title: str,
    outpath: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df[x], df[y1], marker="o")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df[x], df[y2], marker="s")
    ax2.set_ylabel(y2label)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    pivot = df.pivot(index=y, columns=x, values=z)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=z)
    plt.xticks(range(len(pivot.columns)), [f"{v:.0f}" for v in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{v:.0f}" for v in pivot.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_bar_plot(
    df: pd.DataFrame,
    x: str,
    ycols: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    plot_df = df.set_index(x)[ycols]
    plot_df.plot(kind="bar", figsize=(9, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

"""Shared matplotlib styling for TOF plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

COLORS = {
    "signal": "#2563eb",
    "signal_fill": "#dbeafe",
    "theory": "#059669",
    "theory_light": "#a7f3d0",
    "anchor": "#dc2626",
    "dip": "#64748b",
    "grid": "#e2e8f0",
    "text": "#1e293b",
    "muted": "#64748b",
}

ION_GROUP_COLORS = {
    "Argon": "#059669",
    "Methane": "#7c3aed",
    "Holmium": "#d97706",
    "Restgas": "#e11d48",
    "Misc": "#64748b",
    "Custom": "#9333ea",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titleweight": "600",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "legend.framealpha": 0.92,
            "legend.edgecolor": COLORS["grid"],
        }
    )

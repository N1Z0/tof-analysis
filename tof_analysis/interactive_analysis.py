"""Plotting helpers for calibrated spectrum analysis."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from tof_analysis.calibration import Calibration
from tof_analysis.detection import Dip, detect_dips
from tof_analysis.io import Spectrum
from tof_analysis.ion_config import IonDefinition, full_ion_registry
from tof_analysis.plot_style import COLORS, apply_plot_style


def plot_spectrum_with_lines(
    spectrum: Spectrum,
    calibration: Calibration,
    ions: Iterable[IonDefinition],
    *,
    dips: list[Dip] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    t_min_us: float | None = None,
    t_max_us: float | None = None,
    show_labels: bool = True,
    figsize: tuple[float, float] = (18.0, 7.0),
    dpi: int = 120,
) -> plt.Axes:
    """Plot a trace with theoretical flight-time lines from a saved calibration."""
    apply_plot_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    t = spectrum.time_us
    y = spectrum.signal_mV
    if t_min_us is not None:
        mask = t >= t_min_us
        t, y = t[mask], y[mask]
    if t_max_us is not None:
        mask = t <= t_max_us
        t, y = t[mask], y[mask]

    ax.plot(t, y, color=COLORS["signal"], lw=0.85, alpha=0.92, label="MCP signal", zorder=1)

    if dips:
        for dip in dips:
            if t_min_us is not None and dip.time_us < t_min_us:
                continue
            if t_max_us is not None and dip.time_us > t_max_us:
                continue
            ax.axvline(dip.time_us, color=COLORS["dip"], ls="--", lw=0.9, alpha=0.45, zorder=2)

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    label_levels: dict[str, int] = {}
    t_view_min = float(t.min()) if len(t) else 0.0
    t_view_max = float(t.max()) if len(t) else 0.0

    for ion in ions:
        t_line = calibration.predict_time_us(ion.species)
        if t_min_us is not None and t_line < t_min_us:
            continue
        if t_max_us is not None and t_line > t_max_us:
            continue
        if t_line < t_view_min or t_line > t_view_max:
            continue

        color = ion.color
        ax.axvline(t_line, color=color, ls=":", lw=1.35, alpha=0.88, zorder=3)

        if show_labels:
            level = label_levels.get(ion.group, 0)
            label_levels[ion.group] = level + 1
            ax.text(
                t_line,
                ymax - (0.06 + 0.07 * (level % 6)) * span,
                ion.name,
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.88, lw=0.6),
                zorder=4,
            )

    cal = calibration
    subtitle = (
        f"t₀ = {cal.t0_us:.3f} µs · k = {cal.k_us_per_sqrt_mz:.3f} · "
        f"L_eff = {cal.effective_length_m()*100:.1f} cm"
    )
    ax.set(
        xlabel="Oscilloscope time (µs)",
        ylabel="Signal (mV)",
        title=title or f"{spectrum.meta.label}\n{subtitle}",
    )
    ax.grid(True, alpha=0.45)
    ax.set_xlim(left=t_view_min - 0.2, right=t_view_max + 0.2 if len(t) else None)
    if fig is not None:
        fig.set_dpi(dpi)
    return ax


def detect_spectrum_dips(
    spectrum: Spectrum,
    *,
    smooth_sigma: float = 8.0,
    min_prominence_mV: float = 0.3,
    exclude_before_us: float = 3.0,
) -> list[Dip]:
    return detect_dips(
        spectrum.time_us,
        spectrum.signal_mV,
        smooth_sigma=smooth_sigma,
        min_prominence_mV=min_prominence_mV,
        exclude_before_us=exclude_before_us,
    )


def resolve_ion_definitions(names: list[str], registry: dict[str, IonDefinition] | None = None) -> list[IonDefinition]:
    registry = registry or full_ion_registry()
    return [registry[n] for n in names if n in registry]

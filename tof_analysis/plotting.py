"""Plotting helpers for TOF spectra, dips, and calibration overlays."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from tof_analysis.calibration import Assignment, Calibration
from tof_analysis.calibration_store import DEFAULT_CALIBRATION_DIR, load_saved_calibration
from tof_analysis.catalog import SpectrumCatalog
from tof_analysis.detection import Dip
from tof_analysis.io import Spectrum, SpectrumMeta, load_spectrum
from tof_analysis.ion_config import IonDefinition, full_ion_registry
from tof_analysis.physics import IonSpecies
from tof_analysis.plot_style import COLORS, apply_plot_style


def format_dwell_label(meta: SpectrumMeta) -> str:
    """Human-readable integration-time label from metadata or filename."""
    if meta.dwell_us is not None:
        dwell = meta.dwell_us
        if dwell >= 1000:
            return f"{dwell / 1000:g} ms"
        return f"{dwell:g} µs"
    tail = meta.label.split("_")[-1]
    return tail.replace("us", "µs").replace("µs", "µs")


def find_spectra_by_label_pattern(
    data_dir: Path | str,
    pattern: str,
    *,
    gas: bool | None = None,
    exclude_substrings: tuple[str, ...] = ("no_gas",),
) -> list[tuple[SpectrumMeta, Spectrum]]:
    """Return spectra whose label matches a glob pattern (e.g. '*GAS_LASER_ON_45%*')."""
    catalog = SpectrumCatalog.from_directory(data_dir)
    token = pattern.lower()
    matches: list[SpectrumMeta] = []
    for meta in catalog.entries:
        label = meta.label.lower()
        if not fnmatch.fnmatch(label, token):
            continue
        if any(x in label for x in exclude_substrings):
            continue
        if gas is not None and meta.gas is not gas:
            continue
        matches.append(meta)
    matches.sort(key=lambda meta: (meta.dwell_us is None, meta.dwell_us or 0.0, meta.label))
    return [(meta, load_spectrum(meta.path)) for meta in matches]


def default_dwell_overlay_ions(
    registry: dict[str, IonDefinition] | None = None,
) -> list[IonDefinition]:
    """Argon + Holmium ladders, methane q=1 fragments, H+, and D+."""
    registry = registry or full_ion_registry()
    ions: list[IonDefinition] = []
    for group in ("Argon", "Holmium"):
        ions.extend(
            sorted(
                (ion for ion in registry.values() if ion.group == group),
                key=lambda ion: ion.charge,
            )
        )
    ions.extend(
        sorted(
            (ion for ion in registry.values() if ion.group == "Methane" and ion.charge == 1),
            key=lambda ion: ion.mass_amu,
        )
    )
    for name in ("H+", "D+"):
        if name in registry:
            ions.append(registry[name])
    return ions


def _resolve_calibration(
    calibration: Calibration | Path | str | None,
    calibration_dir: Path | str,
) -> Calibration | None:
    if calibration is None:
        return None
    if isinstance(calibration, Calibration):
        return calibration
    path = Path(calibration)
    if not path.is_absolute() and not path.exists():
        path = Path(calibration_dir) / path
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    return load_saved_calibration(path).calibration


def draw_ion_reference_lines(
    ax: plt.Axes,
    calibration: Calibration,
    ions: Iterable[IonDefinition],
    *,
    show_labels: bool = True,
) -> None:
    """Draw calibrated flight-time lines with per-group colours."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    t_left, t_right = ax.get_xlim()
    label_levels: dict[str, int] = {}

    for ion in ions:
        t_line = calibration.predict_time_us(ion.species)
        if t_line < t_left or t_line > t_right:
            continue
        ax.axvline(t_line, color=ion.color, ls=":", lw=1.25, alpha=0.9, zorder=5)
        if not show_labels:
            continue
        level = label_levels.get(ion.group, 0)
        label_levels[ion.group] = level + 1
        ax.text(
            t_line,
            ymax - (0.04 + 0.055 * (level % 8)) * span,
            ion.name,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=ion.color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=ion.color, alpha=0.9, lw=0.5),
            zorder=6,
        )


def plot_dwell_series(
    data_dir: Path | str,
    label_pattern: str = "*GAS_LASER_ON_45%*",
    *,
    gas: bool | None = True,
    calibration: Calibration | Path | str | None = "reference",
    calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR,
    overlay_ions: Iterable[IonDefinition] | None = None,
    show_ion_labels: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (14.0, 6.5),
    dpi: int = 120,
    show: bool = False,
) -> plt.Axes:
    """Overlay spectra that share a filename stem but differ in integration time."""
    apply_plot_style()
    pairs = find_spectra_by_label_pattern(data_dir, label_pattern, gas=gas)
    if not pairs:
        raise ValueError(f"No spectra match pattern {label_pattern!r} in {data_dir}")

    cal = _resolve_calibration(calibration, calibration_dir)
    ions = list(overlay_ions) if overlay_ions is not None else default_dwell_overlay_ions()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    cmap = plt.cm.plasma(np.linspace(0.12, 0.88, len(pairs)))
    for color, (meta, spectrum) in zip(cmap, pairs):
        label = format_dwell_label(meta)
        ax.plot(
            spectrum.time_us,
            spectrum.signal_mV,
            color=color,
            lw=0.9,
            alpha=0.9,
            label=label,
            zorder=2,
        )

    if cal is not None and ions:
        draw_ion_reference_lines(ax, cal, ions, show_labels=show_ion_labels)

    plot_title = title or f"Dwell comparison — {label_pattern}"
    if cal is not None:
        plot_title += (
            f"\nt₀ = {cal.t0_us:.3f} µs · k = {cal.k_us_per_sqrt_mz:.3f} · "
            f"dotted lines: Ar, Ho, CH₄ fragments, H⁺, D⁺"
        )
    ax.set(xlabel="Oscilloscope time (µs)", ylabel="Signal (mV)")
    ax.set_title(plot_title)
    ax.grid(True, alpha=0.35)
    ax.legend(title="Integration time", loc="upper right", fontsize=9, title_fontsize=9)
    if fig is not None:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
    return ax


def plot_spectrum(
    spectrum: Spectrum,
    *,
    ax: plt.Axes | None = None,
    dips: Iterable[Dip] | None = None,
    calibration: Calibration | None = None,
    overlay_ions: Iterable[IonSpecies] | None = None,
    assignments: Iterable[Assignment] | None = None,
    title: str | None = None,
    show: bool = False,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    ax.plot(spectrum.time_us, spectrum.signal_mV, color="#2b6cb0", lw=0.6, alpha=0.85, label="signal")

    if dips:
        for dip in dips:
            ax.axvline(dip.time_us, color="#718096", ls="--", lw=0.8, alpha=0.5)

    label_map: dict[float, str] = {}
    if assignments:
        for item in assignments:
            label_map[item.dip.time_us] = item.ion.name
            ax.axvline(item.dip.time_us, color="#c53030", ls="-", lw=1.2, alpha=0.8)
            ax.scatter([item.dip.time_us], [item.dip.depth_mV], color="#c53030", s=20, zorder=5)

    if calibration and overlay_ions:
        ymax = float(np.percentile(spectrum.signal_mV, 99))
        ymin = float(np.percentile(spectrum.signal_mV, 1))
        span = ymax - ymin
        for idx, ion in enumerate(overlay_ions):
            t = calibration.predict_time_us(ion)
            name = label_map.get(t, ion.name)
            ax.axvline(t, color="#2f855a", ls=":", lw=1.4, alpha=0.9)
            ax.text(
                t,
                ymax - 0.05 * span - (idx % 5) * 0.08 * span,
                name,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#22543d",
            )

    ax.set_xlabel("time (µs)")
    ax.set_ylabel("signal (mV)")
    ax.set_title(title or spectrum.meta.label)
    ax.grid(True, alpha=0.25)
    if show:
        plt.show()
    return ax


def plot_calibration_fit(
    calibration: Calibration,
    assignments: Iterable[Assignment],
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    xs = np.array([a.ion.sqrt_mz for a in assignments])
    ys = np.array([a.dip.time_us for a in assignments])
    ax.scatter(xs, ys, color="#2b6cb0", zorder=3)
    for item in assignments:
        ax.annotate(item.ion.name, (item.ion.sqrt_mz, item.dip.time_us), xytext=(4, 4), textcoords="offset points", fontsize=8)

    grid = np.linspace(max(0.1, xs.min() * 0.8), xs.max() * 1.1, 100)
    ax.plot(grid, calibration.k_us_per_sqrt_mz * grid + calibration.t0_us, color="#c53030", lw=1.5)
    ax.set_xlabel("sqrt(m/z)")
    ax.set_ylabel("flight time (µs)")
    ax.set_title(
        f"Calibration: L≈{calibration.effective_length_m()*100:.1f} cm, "
        f"t0={calibration.t0_us:.3f} µs, RMSE={calibration.rmse_us:.3f} µs"
    )
    ax.grid(True, alpha=0.25)
    if show:
        plt.show()
    return ax


def plot_catalog_overview(catalog_frame, *, ax: plt.Axes | None = None, show: bool = False):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    frame = catalog_frame.dropna(subset=["dwell_us"])
    colors = frame["laser_on"].map({True: "#3182ce", False: "#a0aec0", None: "#718096"})
    ax.scatter(frame["dwell_us"], frame["shot"], c=colors, s=35)
    ax.set_xlabel("integration time (µs)")
    ax.set_ylabel("shot number")
    ax.set_xscale("log")
    ax.set_title("Spectrum catalog (blue=laser on, gray=laser off)")
    ax.grid(True, alpha=0.25)
    if show:
        plt.show()
    return ax

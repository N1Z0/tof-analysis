"""Interactive peak assignment and calibration helpers for notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tof_analysis.calibration import Assignment, Calibration, fit_calibration
from tof_analysis.detection import Dip, detect_dips
from tof_analysis.io import Spectrum, load_spectrum
from tof_analysis.ion_config import flat_ion_options as _flat_ion_options, full_ion_registry
from tof_analysis.physics import IonSpecies, argon_ions, holmium_ions, methane_fragments
from tof_analysis.subtract import subtract_spectra


@dataclass
class PeakAssignment:
    dip: Dip
    ion_name: str = "—"
    use_in_fit: bool = False

    @property
    def ion(self) -> IonSpecies | None:
        if self.ion_name in ("—", "", "none", "None"):
            return None
        reg = full_ion_registry()
        if self.ion_name in reg:
            return reg[self.ion_name].species
        return None


def ion_catalog() -> dict[str, IonSpecies]:
    return {name: d.species for name, d in full_ion_registry().items()}


def ion_names_grouped() -> list[tuple[str, list[str]]]:
    reg = full_ion_registry()
    grouped: dict[str, list[str]] = {}
    for name, ion in reg.items():
        grouped.setdefault(ion.group, []).append(name)
    order = ["Argon", "Methane", "Holmium", "Restgas", "Custom", "Misc"]
    rows = [("—", ["—"])]
    for group in order:
        if group in grouped:
            rows.append((group, sorted(grouped[group])))
    for group in sorted(set(grouped) - set(order)):
        rows.append((group, sorted(grouped[group])))
    return rows


def flat_ion_options() -> list[str]:
    return _flat_ion_options()


def load_calibration_spectrum(
    *,
    data_dir: Path | str,
    gas_file: str,
    background_file: str | None = None,
    use_difference: bool = True,
) -> Spectrum:
    data_dir = Path(data_dir)
    gas = load_spectrum(data_dir / gas_file)
    if use_difference and background_file:
        background = load_spectrum(data_dir / background_file)
        return subtract_spectra(gas, background)
    return gas


def detect_peaks(
    spectrum: Spectrum,
    *,
    smooth_sigma: float = 6.0,
    min_prominence_mV: float = 1.0,
    exclude_before_us: float = 3.0,
    max_dips: int = 25,
) -> list[Dip]:
    return detect_dips(
        spectrum.time_us,
        spectrum.signal_mV,
        smooth_sigma=smooth_sigma,
        min_prominence_mV=min_prominence_mV,
        exclude_before_us=exclude_before_us,
        max_dips=max_dips,
    )


def preset_assignments(
    dips: list[Dip],
    preset: str,
    *,
    t_window: tuple[float, float] = (5.0, 10.5),
    length_m: float = 0.50,
    voltage_V: float = 300.0,
) -> list[PeakAssignment]:
    """Apply a charge-state ladder to the best matching dips in the Ar time window."""
    from itertools import combinations

    presets: dict[str, list[str]] = {
        "Ar6–Ar3": [f"Ar{q}+" for q in (6, 5, 4, 3)],
        "Ar5–Ar2": [f"Ar{q}+" for q in (5, 4, 3, 2)],
        "Ar4–Ar1": [f"Ar{q}+" for q in (4, 3, 2, 1)],
    }
    ion_names = presets.get(preset)
    ordered_all = sorted(dips, key=lambda d: d.time_us)

    if not ion_names:
        return [PeakAssignment(dip=dip) for dip in ordered_all]

    in_window = [d for d in dips if t_window[0] <= d.time_us <= t_window[1]]
    if in_window:
        prom_cut = max(2.0, 0.15 * max(d.prominence_mV for d in in_window))
        in_window = [d for d in in_window if d.prominence_mV >= prom_cut]
    n = len(ion_names)

    best_combo: tuple[Dip, ...] | None = None
    best_rmse = float("inf")
    if len(in_window) >= n:
        for combo in combinations(in_window, n):
            times = [d.time_us for d in combo]
            if any(times[i + 1] - times[i] < 0.08 for i in range(n - 1)):
                continue
            pairs = [
                (d, IonSpecies(ion_name, 39.948, int(ion_name[2:-1])))
                for d, ion_name in zip(sorted(combo, key=lambda d: d.time_us), ion_names)
            ]
            cal = fit_calibration(pairs, voltage_V=voltage_V, length_m=length_m)
            if cal.rmse_us < best_rmse:
                best_rmse = cal.rmse_us
                best_combo = combo

    assign_map: dict[float, str] = {}
    if best_combo:
        for ion_name, dip in zip(ion_names, sorted(best_combo, key=lambda d: d.time_us)):
            assign_map[dip.time_us] = ion_name

    return [
        PeakAssignment(
            dip=dip,
            ion_name=assign_map.get(dip.time_us, "—"),
            use_in_fit=dip.time_us in assign_map,
        )
        for dip in ordered_all
    ]


def assignments_from_anchors(dips: list[Dip], anchors: list[tuple[str, float]]) -> list[PeakAssignment]:
    """Match saved anchor times to nearest detected dips."""
    reg = full_ion_registry()
    rows = [PeakAssignment(dip=dip) for dip in sorted(dips, key=lambda d: d.time_us)]
    for ion_name, target_t in anchors:
        if ion_name not in reg:
            continue
        nearest = min(rows, key=lambda r: abs(r.dip.time_us - target_t))
        nearest.ion_name = ion_name
        nearest.use_in_fit = True
    return rows


def fit_from_assignments(
    assignments: list[PeakAssignment],
    *,
    voltage_V: float = 300.0,
    length_m: float | None = 0.50,
) -> tuple[Calibration, list[Assignment]]:
    pairs: list[tuple[Dip, IonSpecies]] = []
    for row in assignments:
        if not row.use_in_fit or row.ion is None:
            continue
        pairs.append((row.dip, row.ion))
    if len(pairs) < 2:
        raise ValueError("Select at least two peaks with ion assignments for fitting.")

    cal = fit_calibration(pairs, voltage_V=voltage_V, length_m=length_m)
    fitted = [
        Assignment(
            dip=dip,
            ion=ion,
            residual_us=dip.time_us - cal.predict_time_us(ion),
        )
        for dip, ion in pairs
    ]
    return cal, fitted


def assignment_summary(assignments: list[PeakAssignment], calibration: Calibration | None) -> str:
    lines = ["idx   use  time (µs)  ion        residual"]
    for idx, row in enumerate(sorted(assignments, key=lambda r: r.dip.time_us)):
        res = ""
        if calibration and row.use_in_fit and row.ion is not None:
            res = f"{row.dip.time_us - calibration.predict_time_us(row.ion):+.3f} µs"
        lines.append(
            f"[{idx:2d}]  {'Y' if row.use_in_fit else 'n'}   {row.dip.time_us:8.3f}  "
            f"{row.ion_name:10s} {res}"
        )
    if calibration:
        lines.append("")
        lines.append(
            f"t0 = {calibration.t0_us:.4f} µs  RMSE = {calibration.rmse_us:.4f} µs  "
            f"L = {calibration.length_m:.2f} m  V = {calibration.voltage_V:.0f} V"
        )
    return "\n".join(lines)


def plot_detected_peaks(
    spectrum: Spectrum,
    assignments: list[PeakAssignment],
    *,
    calibration: Calibration | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Show spectrum with numbered detected dips for visual inspection."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    t = spectrum.time_us
    y = spectrum.signal_mV
    ax.plot(t, y, color="#2b6cb0", lw=0.7, alpha=0.85, label="signal")

    ordered = sorted(assignments, key=lambda r: r.dip.time_us)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = ymax - ymin if ymax > ymin else 1.0
    label_y = ymax - 0.06 * span

    for idx, row in enumerate(ordered):
        dip = row.dip
        peak_idx = dip.index if 0 <= dip.index < len(y) else int(np.argmin(np.abs(t - dip.time_us)))
        peak_y = float(y[peak_idx])

        if row.use_in_fit:
            color = "#c53030"
            ls = "-"
            lw = 2.0
            alpha = 0.9
        else:
            color = "#718096"
            ls = "--"
            lw = 1.0
            alpha = 0.55

        ax.axvline(dip.time_us, color=color, ls=ls, lw=lw, alpha=alpha)
        ax.scatter([dip.time_us], [peak_y], color=color, s=45, zorder=5, edgecolors="white", linewidths=0.8)

        label = f"[{idx}]\n{dip.time_us:.2f} µs"
        if row.ion_name != "—":
            label = f"[{idx}] {row.ion_name}\n{dip.time_us:.2f} µs"

        ax.annotate(
            label,
            xy=(dip.time_us, peak_y),
            xytext=(0, -18 if peak_y > 0.5 * (ymin + ymax) else 18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
        )

    if calibration:
        for row in ordered:
            if row.use_in_fit and row.ion is not None:
                t_pred = calibration.predict_time_us(row.ion)
                ax.axvline(t_pred, color="#2f855a", ls=":", lw=1.2, alpha=0.75)

    n_fit = sum(1 for r in ordered if r.use_in_fit)
    ax.set(
        xlabel="time (µs)",
        ylabel="signal (mV)",
        title=title or f"{spectrum.meta.label} — {len(ordered)} dips detected, {n_fit} selected for fit",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_assignments(
    spectrum: Spectrum,
    assignments: list[PeakAssignment],
    calibration: Calibration | None = None,
    *,
    overlay_ions: list[IonSpecies] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(spectrum.time_us, spectrum.signal_mV, color="#2b6cb0", lw=0.6, alpha=0.9)

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    if calibration and overlay_ions:
        for ion in overlay_ions:
            t = calibration.predict_time_us(ion)
            ax.axvline(t, color="#a0aec0", ls=":", lw=0.8, alpha=0.6)

    for row in assignments:
        t = row.dip.time_us
        if row.use_in_fit and row.ion is not None:
            ax.axvline(t, color="#c53030", lw=2.0, alpha=0.85)
            label_y = ymax - 0.08 * span
            ax.text(
                t,
                label_y,
                f"{row.ion_name}\n{t:.2f} µs",
                ha="center",
                fontsize=8,
                color="#9b2c2c",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
            )
            if calibration:
                t_pred = calibration.predict_time_us(row.ion)
                if abs(t_pred - t) > 0.03:
                    ax.axvline(t_pred, color="#2f855a", ls="--", lw=1.0, alpha=0.7)
        else:
            ax.axvline(t, color="#718096", ls="--", lw=0.8, alpha=0.35)

    ax.set(xlabel="time (µs)", ylabel="signal (mV)", title=title or spectrum.meta.label)
    ax.grid(True, alpha=0.25)
    return ax


def save_calibration_json(
    calibration: Calibration,
    assignments: list[PeakAssignment],
    *,
    path: Path | str,
    gas_file: str,
    background_file: str | None = None,
) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anchors = [
        {
            "ion": row.ion.name,
            "mass_amu": row.ion.mass_amu,
            "charge": row.ion.charge,
            "time_us": round(row.dip.time_us, 3),
        }
        for row in assignments
        if row.use_in_fit and row.ion is not None
    ]
    payload = {
        "length_m": calibration.length_m,
        "voltage_V": calibration.voltage_V,
        "t0_us": round(calibration.t0_us, 4),
        "k_us_per_sqrt_mz": round(calibration.k_us_per_sqrt_mz, 4),
        "rmse_us": round(calibration.rmse_us, 4),
        "reference_spectrum": gas_file,
        "background_spectrum": background_file,
        "anchors": anchors,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path

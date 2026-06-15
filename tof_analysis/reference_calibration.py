"""Reference TOF calibration derived from Ar/CH4 gas measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tof_analysis.calibration import Calibration, fit_calibration
from tof_analysis.detection import Dip, detect_dips
from tof_analysis.io import Spectrum, load_spectrum
from tof_analysis.physics import IonSpecies, argon_ions, methane_fragments
from tof_analysis.subtract import subtract_spectra

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "calibration" / "reference.json"

GAS_CALIBRATION_FILE = "2026_06_23_013_TOF150eV_GAS_5000µs.CSV"
BACKGROUND_FILE = "2026_06_23_030_TOF150eV_NO_GAS_LASER_OFF_5ms.CSV"

# Four dominant Ar dips in the gas-background spectrum, assigned to Ar5+..Ar2+.
ARGON_CHARGE_STATES = (5, 4, 3, 2)


@dataclass(frozen=True)
class ReferenceCalibrationResult:
    calibration: Calibration
    anchors: list[tuple[IonSpecies, float]]
    gas_file: str
    background_file: str
    diff_spectrum: Spectrum


def _dip(t_us: float) -> Dip:
    return Dip(index=0, time_us=t_us, depth_mV=0.0, prominence_mV=1.0, width_us=0.1)


def detect_argon_dips(
    diff: Spectrum,
    *,
    t_min_us: float = 5.0,
    t_max_us: float = 10.5,
    min_prominence_mV: float = 1.0,
    n_peaks: int = 4,
    argon_charges: tuple[int, ...] = ARGON_CHARGE_STATES,
    length_m: float = 0.50,
    voltage_V: float = 300.0,
) -> list[float]:
    """
    Find the four dips in the subtracted spectrum that best match an Ar charge ladder.

    Prominent dips are collected in the Ar time window, then the subset of four
    increasing times with the lowest fit residual to Ar5+..Ar2+ is chosen.
    """
    from itertools import combinations

    dips = detect_dips(
        diff.time_us,
        diff.signal_mV,
        smooth_sigma=6.0,
        min_prominence_mV=min_prominence_mV,
        exclude_before_us=t_min_us,
        max_dips=30,
    )
    candidates = sorted(
        (d for d in dips if t_min_us <= d.time_us <= t_max_us),
        key=lambda d: d.time_us,
    )
    if len(candidates) < n_peaks:
        raise ValueError(
            f"Found only {len(candidates)} dips in [{t_min_us}, {t_max_us}] µs; "
            "check the gas/background files or detection settings."
        )

    ions = [IonSpecies(f"Ar{q}+", 39.948, q) for q in argon_charges]
    best_times: list[float] | None = None
    best_rmse = np.inf

    for combo in combinations(candidates, n_peaks):
        times = sorted(d.time_us for d in combo)
        if any(times[i + 1] - times[i] < 0.08 for i in range(n_peaks - 1)):
            continue
        pairs = [(_dip(t), ions[i]) for i, t in enumerate(times)]
        cal = fit_calibration(pairs, voltage_V=voltage_V, length_m=length_m)
        if cal.rmse_us < best_rmse:
            best_rmse = cal.rmse_us
            best_times = times

    if best_times is None:
        best_times = sorted(d.time_us for d in candidates[:n_peaks])

    return best_times


def build_reference_calibration(
    *,
    data_dir: Path | str,
    length_m: float = 0.50,
    voltage_V: float = 300.0,
    anchor_times_us: list[float] | None = None,
    argon_charges: tuple[int, ...] = ARGON_CHARGE_STATES,
) -> ReferenceCalibrationResult:
    """
    Build calibration from gas − background spectrum.

    The four clearest Ar dips in the subtracted trace are assigned to
    Ar5+, Ar4+, Ar3+, Ar2+ (charge states most abundant at 150 eV).
    """
    data_dir = Path(data_dir)
    gas = load_spectrum(data_dir / GAS_CALIBRATION_FILE)
    background = load_spectrum(data_dir / BACKGROUND_FILE)
    diff = subtract_spectra(gas, background)

    if anchor_times_us is None:
        anchor_times_us = detect_argon_dips(diff)

    if len(anchor_times_us) != len(argon_charges):
        raise ValueError("Number of dip times must match number of Ar charge states")

    pairs: list[tuple[Dip, IonSpecies]] = []
    parsed: list[tuple[IonSpecies, float]] = []
    for charge, time_us in zip(argon_charges, anchor_times_us):
        ion = IonSpecies(f"Ar{charge}+", 39.948, charge)
        pairs.append((_dip(time_us), ion))
        parsed.append((ion, time_us))

    calibration = fit_calibration(pairs, voltage_V=voltage_V, length_m=length_m)
    return ReferenceCalibrationResult(
        calibration=calibration,
        anchors=parsed,
        gas_file=GAS_CALIBRATION_FILE,
        background_file=BACKGROUND_FILE,
        diff_spectrum=diff,
    )


def save_reference_calibration(
    result: ReferenceCalibrationResult,
    path: Path | str = DEFAULT_REFERENCE_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "length_m": result.calibration.length_m,
        "voltage_V": result.calibration.voltage_V,
        "t0_us": round(result.calibration.t0_us, 4),
        "k_us_per_sqrt_mz": round(result.calibration.k_us_per_sqrt_mz, 4),
        "rmse_us": round(result.calibration.rmse_us, 4),
        "reference_spectrum": result.gas_file,
        "background_spectrum": result.background_file,
        "anchors": [
            {"ion": ion.name, "mass_amu": ion.mass_amu, "charge": ion.charge, "time_us": round(time_us, 3)}
            for ion, time_us in result.anchors
        ],
        "notes": {
            "collision_energy_eV": 150,
            "beamline_acceleration_V": 300,
            "drift_length_m": 0.5,
            "description": (
                "All four anchors from gas − no-gas spectrum. "
                "Dips assigned to Ar5+..Ar2+. Filename TOF150eV is collision energy, not drift voltage."
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_reference_calibration(path: Path | str = DEFAULT_REFERENCE_PATH) -> Calibration:
    path = Path(path)
    data = json.loads(path.read_text())
    return Calibration(
        length_m=float(data["length_m"]),
        voltage_V=float(data["voltage_V"]),
        t0_us=float(data["t0_us"]),
        k_us_per_sqrt_mz=float(data["k_us_per_sqrt_mz"]),
        rmse_us=float(data["rmse_us"]),
        n_points=len(data.get("anchors", [])),
    )


def validation_table(
    calibration: Calibration,
    *,
    data_dir: Path | str = "DATA",
) -> list[dict]:
    """Compare calibration predictions to dips in the gas-background spectrum."""
    data_dir = Path(data_dir)
    gas = load_spectrum(data_dir / GAS_CALIBRATION_FILE)
    background = load_spectrum(data_dir / BACKGROUND_FILE)
    diff = subtract_spectra(gas, background)
    dips = detect_dips(diff.time_us, diff.signal_mV, min_prominence_mV=0.3, exclude_before_us=3.0)

    from tof_analysis.physics import holmium_ions

    candidates = argon_ions(max_charge=8) + methane_fragments() + holmium_ions(max_charge=3)

    rows: list[dict] = []
    for ion in candidates:
        pred = calibration.predict_time_us(ion)
        if not dips:
            nearest = None
            delta = None
        else:
            nearest_dip = min(dips, key=lambda d: abs(d.time_us - pred))
            nearest = nearest_dip.time_us
            delta = nearest - pred
        rows.append(
            {
                "ion": ion.name,
                "sqrt_mz": round(ion.sqrt_mz, 3),
                "predicted_us": round(pred, 3),
                "nearest_dip_us": None if nearest is None else round(nearest, 3),
                "delta_us": None if delta is None else round(delta, 3),
            }
        )
    return rows


def plot_reference_calibration(
    result: ReferenceCalibrationResult,
    *,
    data_dir: Path | str = "DATA",
    output_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot gas−background spectrum with anchor dips and calibrated Ar lines."""
    data_dir = Path(data_dir)
    cal = result.calibration
    diff = result.diff_spectrum
    gas = load_spectrum(data_dir / result.gas_file)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, spec, title in [
        (axes[0], diff, "Gas − no gas (calibration spectrum)"),
        (axes[1], gas, "Raw gas (Ar + CH4)"),
    ]:
        ax.plot(spec.time_us, spec.signal_mV, lw=0.6, color="#2b6cb0")
        ymin, ymax = ax.get_ylim()

        for ion, t_obs in result.anchors:
            t_pred = cal.predict_time_us(ion)
            ax.axvline(t_obs, color="#c53030", lw=2.0, alpha=0.9)
            ax.scatter([t_obs], [ymin], color="#c53030", s=40, zorder=5)
            ax.text(
                t_obs,
                ymax * 0.92,
                f"{ion.name}\n{t_obs:.2f} µs",
                ha="center",
                fontsize=8,
                color="#9b2c2c",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )
            if abs(t_pred - t_obs) > 0.02:
                ax.axvline(t_pred, color="#2f855a", ls=":", lw=1.2, alpha=0.8)

        for ion in argon_ions(max_charge=8):
            if any(a[0].name == ion.name for a in result.anchors):
                continue
            t = cal.predict_time_us(ion)
            if t < spec.time_us.min() or t > spec.time_us.max():
                continue
            ax.axvline(t, color="#718096", ls=":", lw=0.8, alpha=0.5)

        ax.set_ylabel("signal (mV)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[1].set_xlabel("time (µs)")
    fig.suptitle(
        f"Reference calibration: t0 = {cal.t0_us:.3f} µs, RMSE = {cal.rmse_us:.3f} µs "
        f"(L = {cal.length_m:.2f} m, V = {cal.voltage_V:.0f} V)",
        y=1.01,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig

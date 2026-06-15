"""Load and list saved calibration JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tof_analysis.calibration import Calibration

DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"


@dataclass(frozen=True)
class SavedCalibration:
    path: Path
    calibration: Calibration
    anchors: list[dict]
    reference_spectrum: str | None
    background_spectrum: str | None
    notes: dict | str | None

    @property
    def label(self) -> str:
        return self.path.stem

    @property
    def summary(self) -> str:
        cal = self.calibration
        leff = cal.effective_length_m()
        return (
            f"{self.label}:  t₀ = {cal.t0_us:.3f} µs,  k = {cal.k_us_per_sqrt_mz:.3f},  "
            f"L_eff = {leff*100:.1f} cm,  RMSE = {cal.rmse_us:.3f} µs"
        )


def list_calibration_files(directory: Path | str = DEFAULT_CALIBRATION_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def load_saved_calibration(path: Path | str) -> SavedCalibration:
    path = Path(path)
    data = json.loads(path.read_text())
    calibration = Calibration(
        length_m=float(data["length_m"]),
        voltage_V=float(data["voltage_V"]),
        t0_us=float(data["t0_us"]),
        k_us_per_sqrt_mz=float(data["k_us_per_sqrt_mz"]),
        rmse_us=float(data["rmse_us"]),
        n_points=len(data.get("anchors", [])),
    )
    return SavedCalibration(
        path=path,
        calibration=calibration,
        anchors=list(data.get("anchors", [])),
        reference_spectrum=data.get("reference_spectrum"),
        background_spectrum=data.get("background_spectrum"),
        notes=data.get("notes"),
    )

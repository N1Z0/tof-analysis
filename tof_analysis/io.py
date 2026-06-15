"""Load MCP time traces and parse acquisition metadata from filenames."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}_\d{2}_\d{2})_(?P<shot>\d{3})_"
    r"TOF(?P<tof_ev>\d+)eV"
    r"(?:_(?P<gas>GAS|NO_GAS))?"
    r"(?:_(?P<laser>LASER_(?:ON|OFF)(?:_\d+%)?))?"
    r"(?:_(?P<dwell>[\d.]+(?:µs|ms|us)))?",
    re.IGNORECASE,
)


def _parse_dwell(raw: str | None) -> float | None:
    if not raw:
        return None
    value = raw.replace("µ", "u").lower()
    if value.endswith("us"):
        return float(value[:-2])
    if value.endswith("ms"):
        return float(value[:-2]) * 1000.0
    return None


@dataclass(frozen=True)
class SpectrumMeta:
    path: Path
    date: str | None = None
    shot: int | None = None
    tof_ev: float | None = None
    gas: bool | None = None
    laser_on: bool | None = None
    laser_power_pct: float | None = None
    dwell_us: float | None = None
    label: str = field(default="")

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class Spectrum:
    meta: SpectrumMeta
    time_s: np.ndarray
    voltage_V: np.ndarray

    @property
    def time_us(self) -> np.ndarray:
        """Oscilloscope time in µs (absolute, not re-zeroed per trace)."""
        return self.time_s * 1e6

    @property
    def time_origin_us(self) -> float:
        """First sample time in µs (scope window start)."""
        return float(self.time_s.min() * 1e6)

    @property
    def signal_mV(self) -> np.ndarray:
        return self.voltage_V * 1e3


def parse_filename(path: Path | str) -> SpectrumMeta:
    path = Path(path)
    match = _FILENAME_RE.search(path.stem)
    if not match:
        return SpectrumMeta(path=path, label=path.stem)

    groups = match.groupdict()
    laser = groups.get("laser") or ""
    laser_on = None
    laser_pct = None
    if laser:
        laser_on = "ON" in laser.upper()
        pct_match = re.search(r"(\d+)%", laser)
        if pct_match:
            laser_pct = float(pct_match.group(1))

    gas_token = groups.get("gas")
    gas = None if gas_token is None else gas_token.upper() == "GAS"

    return SpectrumMeta(
        path=path,
        date=groups.get("date"),
        shot=int(groups["shot"]) if groups.get("shot") else None,
        tof_ev=float(groups["tof_ev"]) if groups.get("tof_ev") else None,
        gas=gas,
        laser_on=laser_on,
        laser_power_pct=laser_pct,
        dwell_us=_parse_dwell(groups.get("dwell")),
        label=path.stem,
    )


def load_spectrum(path: Path | str) -> Spectrum:
    path = Path(path)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    columns = [str(c).strip().lower() for c in df.columns]
    time_col = next(i for i, c in enumerate(columns) if "s" in c and "in" in c)
    volt_col = 1 if time_col == 0 else 0 if time_col == 1 else 1

    time_s = df.iloc[:, time_col].to_numpy(dtype=float)
    voltage_V = df.iloc[:, volt_col].to_numpy(dtype=float)
    return Spectrum(meta=parse_filename(path), time_s=time_s, voltage_V=voltage_V)


def average_spectra(spectra: list[Spectrum]) -> Spectrum:
    if not spectra:
        raise ValueError("Need at least one spectrum to average")
    if len(spectra) == 1:
        return spectra[0]

    t_min = max(spec.time_s.min() for spec in spectra)
    t_max = min(spec.time_s.max() for spec in spectra)
    if t_max <= t_min:
        raise ValueError("Spectra do not overlap in time")

    grid = np.linspace(t_min, t_max, spectra[0].time_s.size)
    stacked = np.vstack([np.interp(grid, spec.time_s, spec.voltage_V) for spec in spectra])
    meta = SpectrumMeta(
        path=spectra[0].meta.path,
        label=f"avg({len(spectra)})",
    )
    return Spectrum(meta=meta, time_s=grid, voltage_V=stacked.mean(axis=0))

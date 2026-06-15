"""Browse and filter spectra by parsed acquisition metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tof_analysis.io import SpectrumMeta, load_spectrum, parse_filename


@dataclass
class SpectrumCatalog:
    data_dir: Path
    entries: list[SpectrumMeta]

    @classmethod
    def from_directory(cls, data_dir: Path | str) -> "SpectrumCatalog":
        data_dir = Path(data_dir)
        paths = sorted(data_dir.glob("*.CSV")) + sorted(data_dir.glob("*.csv"))
        entries = [parse_filename(path) for path in paths]
        return cls(data_dir=data_dir, entries=entries)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for entry in self.entries:
            rows.append(
                {
                    "path": str(entry.path),
                    "label": entry.label,
                    "date": entry.date,
                    "shot": entry.shot,
                    "tof_ev": entry.tof_ev,
                    "gas": entry.gas,
                    "laser_on": entry.laser_on,
                    "laser_power_pct": entry.laser_power_pct,
                    "dwell_us": entry.dwell_us,
                }
            )
        return pd.DataFrame(rows)

    def filter(
        self,
        *,
        gas: bool | None = None,
        laser_on: bool | None = None,
        min_dwell_us: float | None = None,
        max_dwell_us: float | None = None,
        tof_ev: float | None = None,
        label_contains: str | None = None,
    ) -> list[SpectrumMeta]:
        results = self.entries
        if gas is not None:
            results = [e for e in results if e.gas is gas]
        if laser_on is not None:
            results = [e for e in results if e.laser_on is laser_on]
        if tof_ev is not None:
            results = [e for e in results if e.tof_ev == tof_ev]
        if min_dwell_us is not None:
            results = [e for e in results if e.dwell_us is not None and e.dwell_us >= min_dwell_us]
        if max_dwell_us is not None:
            results = [e for e in results if e.dwell_us is not None and e.dwell_us <= max_dwell_us]
        if label_contains:
            token = label_contains.lower()
            results = [e for e in results if token in e.label.lower()]
        return results

    def load(self, meta: SpectrumMeta):
        return load_spectrum(meta.path)

    def load_filtered(self, **kwargs):
        return [self.load(meta) for meta in self.filter(**kwargs)]

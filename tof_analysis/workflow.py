"""High-level analysis workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tof_analysis.calibration import Assignment, Calibration, fit_calibration, suggest_ion_assignments
from tof_analysis.catalog import SpectrumCatalog
from tof_analysis.detection import Dip, detect_dips
from tof_analysis.io import Spectrum, average_spectra, load_spectrum
from tof_analysis.physics import IonSpecies, ion_library


@dataclass
class AnalysisConfig:
    data_dir: Path = Path("DATA")
    length_m: float = 0.50
    voltage_V: float = 300.0
    t0_us_guess: float = 0.074
    smooth_sigma: float = 8.0
    min_prominence_mV: float = 0.3
    exclude_before_us: float = 3.0
    candidate_ions: list[IonSpecies] = field(default_factory=ion_library)
    assignment_prominence_fraction: float = 0.08
    max_assignment_dips: int = 8
    max_assignment_residual_us: float = 1.5


@dataclass
class AnalysisResult:
    spectrum: Spectrum
    dips: list[Dip]
    assignments: list[Assignment]
    calibration: Calibration | None = None


class TofWorkflow:
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.catalog = SpectrumCatalog.from_directory(self.config.data_dir)

    def load_one(self, path: Path | str) -> Spectrum:
        return load_spectrum(path)

    def load_by_label(self, token: str) -> Spectrum:
        matches = [m for m in self.catalog.entries if token.lower() in m.label.lower()]
        if not matches:
            raise FileNotFoundError(f"No spectrum matching '{token}'")
        return load_spectrum(matches[0].path)

    def load_average(self, **filter_kwargs) -> Spectrum:
        metas = self.catalog.filter(**filter_kwargs)
        if not metas:
            raise ValueError("No spectra matched the filter")
        spectra = [load_spectrum(meta.path) for meta in metas]
        return average_spectra(spectra)

    def analyze(
        self,
        spectrum: Spectrum,
        *,
        manual_assignments: list[tuple[Dip, IonSpecies]] | None = None,
        auto_suggest: bool = True,
    ) -> AnalysisResult:
        dips = detect_dips(
            spectrum.time_us,
            spectrum.signal_mV,
            smooth_sigma=self.config.smooth_sigma,
            min_prominence_mV=self.config.min_prominence_mV,
            exclude_before_us=self.config.exclude_before_us,
        )
        voltage = self.config.voltage_V

        significant_dips = dips
        if dips:
            max_prom = max(d.prominence_mV for d in dips)
            threshold = max(
                self.config.min_prominence_mV,
                self.config.assignment_prominence_fraction * max_prom,
            )
            significant_dips = [d for d in dips if d.prominence_mV >= threshold]

        calibration = None
        assignments: list[Assignment] = []
        if manual_assignments:
            calibration = fit_calibration(
                manual_assignments,
                voltage_V=voltage,
                length_m=self.config.length_m,
            )
            assignments = [
                Assignment(
                    dip=dip,
                    ion=ion,
                    residual_us=dip.time_us - calibration.predict_time_us(ion),
                )
                for dip, ion in manual_assignments
            ]
        elif auto_suggest and significant_dips:
            assignments = suggest_ion_assignments(
                significant_dips,
                self.config.candidate_ions,
                voltage_V=voltage,
                length_m=self.config.length_m,
                t0_us=self.config.t0_us_guess,
                max_dips=self.config.max_assignment_dips,
            )
            assignments = [
                a for a in assignments if abs(a.residual_us) <= self.config.max_assignment_residual_us
            ]
            pairs = [(a.dip, a.ion) for a in assignments]
            if len(pairs) >= 2:
                calibration = fit_calibration(
                    pairs,
                    voltage_V=voltage,
                    length_m=None,
                )

        return AnalysisResult(
            spectrum=spectrum,
            dips=dips,
            assignments=assignments,
            calibration=calibration,
        )

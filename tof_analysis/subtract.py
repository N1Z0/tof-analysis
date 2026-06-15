"""Background subtraction for gas-reference spectra."""

from __future__ import annotations

import numpy as np

from tof_analysis.io import Spectrum


def subtract_spectra(foreground: Spectrum, background: Spectrum) -> Spectrum:
    """Align foreground and background in time and return foreground - background."""
    from tof_analysis.io import SpectrumMeta

    t_min = max(foreground.time_s.min(), background.time_s.min())
    t_max = min(foreground.time_s.max(), background.time_s.max())
    if t_max <= t_min:
        raise ValueError("Spectra do not overlap in time")

    grid = np.linspace(t_min, t_max, foreground.time_s.size)
    v_fg = np.interp(grid, foreground.time_s, foreground.voltage_V)
    v_bg = np.interp(grid, background.time_s, background.voltage_V)
    meta = SpectrumMeta(
        path=foreground.meta.path,
        label=f"{foreground.meta.stem} - {background.meta.stem}",
    )
    return Spectrum(meta=meta, time_s=grid, voltage_V=v_fg - v_bg)

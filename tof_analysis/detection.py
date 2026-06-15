"""Detect negative-going ion dips in MCP traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Dip:
    index: int
    time_us: float
    depth_mV: float
    prominence_mV: float
    width_us: float
    fwhm_us: float | None = None


def detect_dips(
    time_us: np.ndarray,
    signal_mV: np.ndarray,
    *,
    smooth_sigma: float = 8.0,
    min_prominence_mV: float = 0.15,
    min_distance_us: float = 0.08,
    exclude_before_us: float = 2.0,
    max_dips: int | None = 20,
) -> list[Dip]:
    """
    Find ion arrival dips (negative peaks) in a smoothed MCP trace.

    Parameters
    ----------
    smooth_sigma:
        Gaussian smoothing width in samples (0.8 ns sampling -> ~6 ns at sigma=8).
    exclude_before_us:
        Ignore the first *exclude_before_us* after the trace start (scope window
        minimum). Works with absolute oscilloscope timestamps.
    """
    if time_us.size < 5:
        return []

    t_start_us = float(np.min(time_us))
    gate_us = t_start_us + exclude_before_us

    smoothed = gaussian_filter1d(signal_mV, sigma=smooth_sigma)
    dt_us = float(np.median(np.diff(time_us)))
    min_distance = max(1, int(round(min_distance_us / dt_us)))

    inverted = -smoothed
    baseline = float(np.percentile(smoothed, 75))
    peaks, props = find_peaks(
        inverted,
        prominence=min_prominence_mV,
        distance=min_distance,
        width=1,
    )

    dips: list[Dip] = []
    for idx, peak_idx in enumerate(peaks):
        if time_us[peak_idx] < gate_us:
            continue
        depth = float(smoothed[peak_idx] - baseline)
        width_samples = float(props["widths"][idx]) if "widths" in props else 1.0
        width_us = width_samples * dt_us
        left = max(0, peak_idx - int(width_samples))
        right = min(len(smoothed) - 1, peak_idx + int(width_samples))
        half_max = baseline + 0.5 * (smoothed[peak_idx] - baseline)
        region = smoothed[left : right + 1]
        above = region <= half_max
        fwhm_us = None
        if above.any():
            fwhm_us = above.sum() * dt_us

        dips.append(
            Dip(
                index=int(peak_idx),
                time_us=float(time_us[peak_idx]),
                depth_mV=depth,
                prominence_mV=float(props["prominences"][idx]),
                width_us=width_us,
                fwhm_us=fwhm_us,
            )
        )

    dips.sort(key=lambda d: d.prominence_mV, reverse=True)
    if max_dips is not None:
        dips = dips[:max_dips]
    dips.sort(key=lambda d: d.time_us)
    return dips

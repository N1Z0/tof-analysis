"""Launch the interactive calibration notebook UI."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from tof_analysis.catalog import SpectrumCatalog
from tof_analysis.interactive_calibration import (
    PeakAssignment,
    assignment_summary,
    assignments_from_anchors,
    detect_peaks,
    fit_from_assignments,
    flat_ion_options,
    plot_assignments,
    plot_detected_peaks,
    preset_assignments,
    save_calibration_json,
)
from tof_analysis.ion_config import sanitize_calibration_name
from tof_analysis.io import load_spectrum
from tof_analysis.physics import argon_ions
from tof_analysis.reference_calibration import load_reference_calibration


class CalibrationWorkbench:
    """Widget-based UI for assigning ions to dips and fitting TOF calibration."""

    def __init__(self, data_dir: Path | str = "DATA"):
        self.data_dir = Path(data_dir)
        self.catalog = SpectrumCatalog.from_directory(self.data_dir)
        self.spectrum = None
        self.dips: list = []
        self.assignments: list[PeakAssignment] = []
        self.calibration = None
        self._assignment_rows: list[tuple[widgets.Checkbox, widgets.Dropdown]] = []
        self._build_widgets()
        self._wire_events()
        self._load_default_spectrum()

    def _build_widgets(self):
        labels = sorted(self.catalog.to_frame()["label"].tolist())
        self.gas_dd = widgets.Dropdown(
            options=labels,
            value=next(l for l in labels if "013" in l and "GAS" in l),
            description="Gas file:",
            layout=widgets.Layout(width="95%"),
        )
        self.bg_dd = widgets.Dropdown(
            options=["(none)"] + labels,
            value=next(l for l in labels if "030" in l and "NO_GAS" in l),
            description="Background:",
            layout=widgets.Layout(width="95%"),
        )
        self.use_diff = widgets.Checkbox(value=True, description="Use gas − background")
        self.smooth = widgets.FloatSlider(value=6, min=1, max=25, step=1, description="Smooth σ")
        self.prom = widgets.FloatSlider(value=1.0, min=0.1, max=10, step=0.1, description="Min prom mV")
        self.gate = widgets.FloatSlider(value=3.0, min=0, max=8, step=0.5, description="Gate before µs")
        self.length = widgets.FloatText(value=0.50, description="L (m)")
        self.voltage = widgets.FloatText(value=300.0, description="V_acc")
        self.fix_length = widgets.Checkbox(value=True, description="Fix L (fit t0 only)")
        self.preset = widgets.Dropdown(
            options=["— manual —", "Ar5–Ar2", "Ar6–Ar3", "Ar4–Ar1", "Load saved"],
            value="Ar5–Ar2",
            description="Preset:",
        )
        self.detect_btn = widgets.Button(description="Detect peaks", button_style="info")
        self.fit_btn = widgets.Button(description="Fit & plot", button_style="primary")
        self.save_btn = widgets.Button(description="Save calibration")
        self.cal_name = widgets.Text(
            value="reference",
            description="Save as:",
            placeholder="e.g. june2026_L_unfixed",
            layout=widgets.Layout(width="95%"),
        )
        self.output = widgets.Output()
        self.summary = widgets.Textarea(
            value="",
            layout=widgets.Layout(width="95%", height="160px"),
            disabled=True,
        )
        self.assignment_box = widgets.VBox([])
        self.peak_figure = widgets.Output()
        self.fit_figure = widgets.Output()

    def _wire_events(self):
        self.detect_btn.on_click(self._on_detect)
        self.fit_btn.on_click(self._on_fit)
        self.save_btn.on_click(self._on_save)
        self.preset.observe(self._on_preset, names="value")
        self.gas_dd.observe(lambda _: self._reload_spectrum(), names="value")
        self.bg_dd.observe(lambda _: self._reload_spectrum(), names="value")
        self.use_diff.observe(lambda _: self._reload_spectrum(), names="value")

    def _resolve_path(self, label: str) -> Path:
        match = next(m for m in self.catalog.entries if m.label == label)
        return match.path

    def _reload_spectrum(self):
        gas_path = self._resolve_path(self.gas_dd.value)
        bg_label = self.bg_dd.value
        bg_path = None if bg_label == "(none)" else self._resolve_path(bg_label)
        if self.use_diff.value and bg_path:
            from tof_analysis.subtract import subtract_spectra

            self.spectrum = subtract_spectra(load_spectrum(gas_path), load_spectrum(bg_path))
        else:
            self.spectrum = load_spectrum(gas_path)
        self.calibration = None
        self._on_detect(None)

    def _load_default_spectrum(self):
        self._reload_spectrum()

    def _read_assignments_from_widgets(self):
        ordered = sorted(self.assignments, key=lambda r: r.dip.time_us)
        for (cb, dd), row in zip(self._assignment_rows, ordered):
            row.use_in_fit = cb.value
            row.ion_name = dd.value

    def _refresh_peak_plot(self):
        if self.spectrum is None or not self.assignments:
            return
        with self.peak_figure:
            self.peak_figure.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_detected_peaks(
                self.spectrum,
                self.assignments,
                calibration=self.calibration,
                ax=ax,
            )
            plt.tight_layout()
            plt.show()

    def _on_assignment_change(self, _=None):
        if not self._assignment_rows:
            return
        self._read_assignments_from_widgets()
        self._refresh_peak_plot()

    def _on_detect(self, _):
        if self.spectrum is None:
            self._load_default_spectrum()
            return
        self.dips = detect_peaks(
            self.spectrum,
            smooth_sigma=self.smooth.value,
            min_prominence_mV=self.prom.value,
            exclude_before_us=self.gate.value,
        )
        preset = self.preset.value
        if preset == "Load saved":
            try:
                import json

                data = json.loads(
                    (self.data_dir.parent / "calibration" / "reference.json").read_text()
                )
                anchors = [(a["ion"], a["time_us"]) for a in data["anchors"]]
                self.assignments = assignments_from_anchors(self.dips, anchors)
            except Exception:
                self.assignments = preset_assignments(self.dips, "Ar5–Ar2")
        elif preset == "— manual —":
            self.assignments = [PeakAssignment(dip=d) for d in sorted(self.dips, key=lambda d: d.time_us)]
        else:
            self.assignments = preset_assignments(self.dips, preset)
        self.calibration = None
        self._rebuild_assignment_table()
        self._refresh_peak_plot()

    def _on_preset(self, change):
        if not self.dips or change["new"] == "— manual —":
            return
        if change["new"] == "Load saved":
            self._on_detect(None)
            return
        self.assignments = preset_assignments(self.dips, change["new"])
        self._sync_widgets_from_assignments()
        self._refresh_peak_plot()

    def _rebuild_assignment_table(self):
        self._assignment_rows.clear()
        rows = []
        ion_options = flat_ion_options()
        header = widgets.HTML(
            "<b>Peak assignments</b> — indices match the plot labels "
            "(<span style='color:#c53030'>red = selected</span>, "
            "<span style='color:#718096'>grey = detected only</span>)"
        )
        for idx, row in enumerate(sorted(self.assignments, key=lambda r: r.dip.time_us)):
            cb = widgets.Checkbox(
                value=row.use_in_fit,
                description=f"[{idx}] {row.dip.time_us:.3f} µs  (prom {row.dip.prominence_mV:.1f} mV)",
            )
            dd = widgets.Dropdown(
                options=ion_options,
                value=row.ion_name if row.ion_name in ion_options else "—",
                layout=widgets.Layout(width="220px"),
            )
            cb.observe(self._on_assignment_change, names="value")
            dd.observe(self._on_assignment_change, names="value")
            self._assignment_rows.append((cb, dd))
            rows.append(widgets.HBox([cb, dd]))
        self.assignment_box.children = [header, *rows]

    def _sync_widgets_from_assignments(self):
        ordered = sorted(self.assignments, key=lambda r: r.dip.time_us)
        for (cb, dd), row in zip(self._assignment_rows, ordered):
            cb.unobserve(self._on_assignment_change, names="value")
            dd.unobserve(self._on_assignment_change, names="value")
            cb.value = row.use_in_fit
            dd.value = row.ion_name if row.ion_name in flat_ion_options() else "—"
            cb.observe(self._on_assignment_change, names="value")
            dd.observe(self._on_assignment_change, names="value")

    def _on_fit(self, _):
        with self.output:
            self.output.clear_output(wait=True)
            if not self._assignment_rows:
                print("Detect peaks first.")
                return
            self._read_assignments_from_widgets()
            try:
                length = self.length.value if self.fix_length.value else None
                self.calibration, _ = fit_from_assignments(
                    self.assignments,
                    voltage_V=self.voltage.value,
                    length_m=length,
                )
            except ValueError as exc:
                print(exc)
                return

            self.summary.value = assignment_summary(self.assignments, self.calibration)
            print(self.summary.value)

            self._refresh_peak_plot()

            overlay = argon_ions(max_charge=8)
            with self.fit_figure:
                self.fit_figure.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(12, 4.5))
                plot_assignments(
                    self.spectrum,
                    self.assignments,
                    self.calibration,
                    overlay_ions=overlay,
                    ax=ax,
                    title=f"Calibration fit — {self.spectrum.meta.label}",
                )
                plt.tight_layout()
                plt.show()

    def _on_save(self, _):
        with self.output:
            if self.calibration is None:
                print("Fit calibration before saving.")
                return
            self._read_assignments_from_widgets()
            gas_file = self._resolve_path(self.gas_dd.value).name
            bg = None if self.bg_dd.value == "(none)" else self._resolve_path(self.bg_dd.value).name
            slug = sanitize_calibration_name(self.cal_name.value)
            cal_dir = self.data_dir.parent / "calibration"
            path = save_calibration_json(
                self.calibration,
                self.assignments,
                path=cal_dir / f"{slug}.json",
                gas_file=gas_file,
                background_file=bg,
            )
            fig_path = cal_dir / f"{slug}_fit.png"
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_detected_peaks(
                self.spectrum,
                self.assignments,
                calibration=self.calibration,
                ax=ax,
                title=f"Saved calibration — {self.spectrum.meta.label}",
            )
            plt.tight_layout()
            fig.savefig(fig_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path} and {fig_path}")
            self.cal_name.value = slug

    def display(self):
        ui = widgets.VBox(
            [
                widgets.HTML(
                    "<h2 style='margin:0 0 4px 0;color:#1e293b'>TOF calibration</h2>"
                    "<p style='margin:0 0 12px 0;color:#64748b;font-size:13px'>"
                    "Detect dips, assign ions, fit t = k√(m/z) + t₀, save to calibration/</p>"
                ),
                widgets.HTML("<h3 style='margin:12px 0 6px 0'>1 · Spectrum</h3>"),
                self.gas_dd,
                self.bg_dd,
                self.use_diff,
                widgets.HTML("<h3 style='margin:16px 0 6px 0'>2 · Detect peaks</h3>"),
                widgets.HBox([self.smooth, self.prom, self.gate]),
                widgets.HBox([self.detect_btn, self.preset]),
                widgets.HTML(
                    "<b>Detected dips on spectrum</b> — numbered [0], [1], … match rows below. "
                    "Adjust sliders and click <i>Detect peaks</i> again if needed."
                ),
                self.peak_figure,
                widgets.HTML("<h3 style='margin:16px 0 6px 0'>3 · Assign (q, m)</h3>"),
                self.assignment_box,
                widgets.HTML("<h3 style='margin:16px 0 6px 0'>4 · Fit & save</h3>"),
                widgets.HBox([self.length, self.voltage, self.fix_length]),
                self.cal_name,
                widgets.HBox([self.fit_btn, self.save_btn]),
                self.summary,
                widgets.HTML("<b>Calibration overlay</b> (after fit)"),
                self.fit_figure,
                self.output,
            ]
        )
        display(ui)


def launch_workbench(data_dir: Path | str = "DATA") -> CalibrationWorkbench:
    bench = CalibrationWorkbench(data_dir=data_dir)
    bench.display()
    return bench

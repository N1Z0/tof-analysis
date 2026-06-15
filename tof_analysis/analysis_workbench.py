"""Interactive spectrum analysis workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from tof_analysis.calibration_store import (
    DEFAULT_CALIBRATION_DIR,
    list_calibration_files,
    load_saved_calibration,
)
from tof_analysis.catalog import SpectrumCatalog
from tof_analysis.interactive_analysis import (
    detect_spectrum_dips,
    plot_spectrum_with_lines,
)
from tof_analysis.io import average_spectra, load_spectrum
from tof_analysis.ion_config import (
    DEFAULT_CONFIG_PATH,
    GROUP_COLORS,
    AnalysisSpecies,
    CustomIonEntry,
    analysis_species_catalog,
    expand_species_selection,
    load_custom_entries,
    save_custom_entries,
)
from tof_analysis.subtract import subtract_spectra
from tof_analysis.workbench_state import (
    DEFAULT_ANALYSIS_STATE_PATH,
    load_analysis_state,
    save_analysis_state,
)

QUICK_GROUPS = ["Argon", "Methane", "Holmium", "Restgas"]
PLOT_FIGSIZE = (18.0, 7.5)
PLOT_DPI = 120
_Q_STYLE = {"description_width": "28px"}
_Q_LAYOUT = widgets.Layout(width="90px", min_width="90px", margin="0 4px 0 0")
_ROW_LAYOUT = widgets.Layout(padding="2px 0 8px 0")


@dataclass
class _SpeciesRow:
    species: AnalysisSpecies
    enabled: widgets.Checkbox
    q_min: widgets.BoundedIntText
    q_max: widgets.BoundedIntText


class AnalysisWorkbench:
    """Browse spectra and overlay theoretical lines from a saved calibration."""

    def __init__(
        self,
        data_dir: Path | str = "DATA",
        calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR,
        ion_config_path: Path | str = DEFAULT_CONFIG_PATH,
        settings_path: Path | str = DEFAULT_ANALYSIS_STATE_PATH,
    ):
        self.data_dir = Path(data_dir)
        self.calibration_dir = Path(calibration_dir)
        self.ion_config_path = Path(ion_config_path)
        self.settings_path = Path(settings_path)
        self.catalog = SpectrumCatalog.from_directory(self.data_dir)
        self.registry = {}
        self.saved_cal = None
        self.spectrum = None
        self._species_rows: dict[str, _SpeciesRow] = {}
        self._loading = True
        self._saved_state = load_analysis_state(self.settings_path)
        self._build_widgets()
        self._wire_events()
        self._refresh_calibration_list()
        self._apply_saved_calibration()
        self._rebuild_species_list()
        self._rebuild_custom_list()
        self._refresh_spectrum_list()
        self._apply_saved_spectrum_selection()
        self._loading = False

    def _reload_registry(self):
        from tof_analysis.ion_config import full_ion_registry

        self.registry = full_ion_registry(self.ion_config_path)

    def _saved(self, key: str, default: Any = None) -> Any:
        return self._saved_state.get(key, default)

    def _build_widgets(self):
        labels = sorted(self.catalog.to_frame()["label"].tolist())
        dwell_opts = ["any"] + sorted(
            self.catalog.to_frame()["dwell_us"].dropna().unique().astype(int).tolist()
        )
        dwell_default = self._saved("dwell", "any")
        if dwell_default not in dwell_opts:
            dwell_default = "any"

        self.cal_dd = widgets.Dropdown(description="Calibration:", layout=widgets.Layout(width="95%"))
        self.cal_info = widgets.HTML(value="")

        self.gas_f = widgets.Dropdown(
            options=["any", "yes", "no"],
            value=self._saved("gas", "any"),
            description="Gas:",
        )
        self.laser_f = widgets.Dropdown(
            options=["any", "on", "off"],
            value=self._saved("laser", "any"),
            description="Laser:",
        )
        self.dwell_f = widgets.Dropdown(
            options=dwell_opts,
            value=dwell_default,
            description="Dwell µs:",
        )
        self.search = widgets.Text(
            value=str(self._saved("search", "")),
            placeholder="Filter label text…",
            description="Search:",
        )
        self.mode = widgets.RadioButtons(
            options=["Single spectrum", "Average filtered"],
            value=self._saved("mode", "Single spectrum"),
            description="Mode:",
        )
        self.spec_dd = widgets.Dropdown(options=labels, description="Spectrum:", layout=widgets.Layout(width="95%"))
        self.bg_dd = widgets.Dropdown(
            options=["(none)"] + labels,
            value=self._saved("background", "(none)"),
            description="Subtract:",
        )
        self.tmin = widgets.FloatText(value=float(self._saved("tmin", 0.0)), description="t min µs (0=all)")
        self.tmax = widgets.FloatText(value=float(self._saved("tmax", 0.0)), description="t max µs (0=all)")

        self.species_panel = widgets.VBox(layout=widgets.Layout(width="99%"))
        self.select_all_ions = widgets.Button(description="Select all", layout=widgets.Layout(width="100px"))
        self.select_none_ions = widgets.Button(description="Clear all", layout=widgets.Layout(width="100px"))
        self.quick_group_buttons = [
            widgets.Button(description=f"+ {g}", layout=widgets.Layout(width="90px")) for g in QUICK_GROUPS
        ]

        self.ci_prefix = widgets.Text(value="Fe", description="Prefix")
        self.ci_mass = widgets.FloatText(value=56.0, description="Mass (amu)")
        self.ci_qmin = widgets.IntText(value=1, description="q min")
        self.ci_qmax = widgets.IntText(value=2, description="q max")
        self.ci_color = widgets.ColorPicker(value=GROUP_COLORS["Custom"], description="Color")
        self.ci_group = widgets.Text(value="Custom", description="Group")
        self.ci_add = widgets.Button(description="Add ion entry", button_style="success")
        self.ci_remove = widgets.Dropdown(description="Remove entry", layout=widgets.Layout(width="95%"))
        self.ci_delete = widgets.Button(description="Delete selected", button_style="warning")
        self.custom_list = widgets.HTML(value="")

        self.show_dips = widgets.Checkbox(
            value=bool(self._saved("show_dips", False)),
            description="Mark detected dips",
        )
        self.plot_btn = widgets.Button(description="Plot spectrum", button_style="primary")
        self.figure = widgets.Output()
        self.output = widgets.Output()

    def _wire_events(self):
        for w in (self.gas_f, self.laser_f, self.dwell_f, self.search, self.mode):
            w.observe(self._on_filter_change, names="value")
        for w in (
            self.cal_dd,
            self.spec_dd,
            self.bg_dd,
            self.tmin,
            self.tmax,
            self.show_dips,
        ):
            w.observe(lambda _: self._persist_state(), names="value")
        self.cal_dd.observe(self._on_calibration_change, names="value")
        self.plot_btn.on_click(self._on_plot)
        self.select_all_ions.on_click(lambda _: self._set_all_species(True))
        self.select_none_ions.on_click(lambda _: self._set_all_species(False))
        for btn, group in zip(self.quick_group_buttons, QUICK_GROUPS):
            btn.on_click(lambda _, g=group: self._select_group(g))
        self.ci_add.on_click(self._on_add_custom_ion)
        self.ci_delete.on_click(self._on_delete_custom_ion)

    def _on_filter_change(self, _):
        self._refresh_spectrum_list()
        self._apply_saved_spectrum_selection()
        self._persist_state()

    def _persist_state(self):
        if self._loading:
            return
        save_analysis_state(self._collect_state(), self.settings_path)

    def _collect_state(self) -> dict[str, Any]:
        cal_name = None
        if isinstance(self.cal_dd.value, Path):
            cal_name = self.cal_dd.value.stem
        species: dict[str, dict[str, Any]] = {}
        for key, row in self._species_rows.items():
            species[key] = {
                "enabled": bool(row.enabled.value),
                "q_min": int(row.q_min.value),
                "q_max": int(row.q_max.value),
            }
        return {
            "calibration": cal_name,
            "gas": self.gas_f.value,
            "laser": self.laser_f.value,
            "dwell": self.dwell_f.value,
            "search": self.search.value,
            "mode": self.mode.value,
            "spectrum": self.spec_dd.value,
            "background": self.bg_dd.value,
            "tmin": float(self.tmin.value),
            "tmax": float(self.tmax.value),
            "show_dips": bool(self.show_dips.value),
            "species": species,
        }

    def _apply_saved_calibration(self):
        name = self._saved("calibration")
        if not name or not isinstance(self.cal_dd.options, list):
            return
        for label, path in self.cal_dd.options:
            if label == name:
                self.cal_dd.value = path
                return

    def _apply_saved_spectrum_selection(self):
        label = self._saved("spectrum")
        if label and label in self.spec_dd.options:
            self.spec_dd.value = label
        bg = self._saved("background", "(none)")
        if bg in self.bg_dd.options:
            self.bg_dd.value = bg

    def _set_all_species(self, value: bool):
        for row in self._species_rows.values():
            row.enabled.value = value
        self._persist_state()

    def _select_group(self, group: str):
        for row in self._species_rows.values():
            if row.species.group == group:
                row.enabled.value = True
        self._persist_state()

    def _species_state(self) -> dict[str, tuple[bool, int, int]]:
        return {
            key: (row.enabled.value, row.q_min.value, row.q_max.value)
            for key, row in self._species_rows.items()
        }

    def _saved_species_row(self, species: AnalysisSpecies) -> tuple[bool, int, int]:
        saved = self._saved("species", {}).get(species.key, {})
        if not saved:
            return False, species.q_min, species.q_max
        q_min = int(saved.get("q_min", species.q_min))
        q_max = int(saved.get("q_max", species.q_max))
        q_min = max(species.q_min, min(q_min, species.q_max))
        q_max = max(species.q_min, min(q_max, species.q_max))
        if q_min > q_max:
            q_min, q_max = species.q_min, species.q_max
        return bool(saved.get("enabled", False)), q_min, q_max

    def _make_q_fields(
        self,
        species: AnalysisSpecies,
        prev_qmin: int,
        prev_qmax: int,
    ) -> tuple[widgets.BoundedIntText, widgets.BoundedIntText]:
        if species.fixed_ion:
            q_min = widgets.BoundedIntText(
                value=species.q_min,
                min=species.q_min,
                max=species.q_max,
                description="min",
                disabled=True,
                style=_Q_STYLE,
                layout=_Q_LAYOUT,
            )
            q_max = widgets.BoundedIntText(
                value=species.q_max,
                min=species.q_min,
                max=species.q_max,
                description="max",
                disabled=True,
                style=_Q_STYLE,
                layout=_Q_LAYOUT,
            )
            return q_min, q_max

        q_min = widgets.BoundedIntText(
            value=max(species.q_min, min(prev_qmin, species.q_max)),
            min=species.q_min,
            max=species.q_max,
            description="min",
            style=_Q_STYLE,
            layout=_Q_LAYOUT,
        )
        q_max = widgets.BoundedIntText(
            value=min(species.q_max, max(prev_qmax, species.q_min)),
            min=species.q_min,
            max=species.q_max,
            description="max",
            style=_Q_STYLE,
            layout=_Q_LAYOUT,
        )

        def _sync_qmin(change, qmax=q_max):
            if change["new"] > qmax.value:
                qmax.value = change["new"]

        def _sync_qmax(change, qmin=q_min):
            if change["new"] < qmin.value:
                qmin.value = change["new"]

        q_min.observe(_sync_qmin, names="value")
        q_max.observe(_sync_qmax, names="value")
        return q_min, q_max

    def _make_species_row(
        self,
        species: AnalysisSpecies,
        prev_on: bool,
        prev_qmin: int,
        prev_qmax: int,
    ) -> tuple[widgets.VBox, _SpeciesRow]:
        enabled = widgets.Checkbox(
            value=prev_on,
            description=species.label,
            indent=False,
            layout=widgets.Layout(width="99%"),
        )
        q_min, q_max = self._make_q_fields(species, prev_qmin, prev_qmax)
        row = widgets.VBox(
            [
                enabled,
                widgets.HBox(
                    [q_min, q_max],
                    layout=widgets.Layout(margin="0 0 0 22px"),
                ),
            ],
            layout=_ROW_LAYOUT,
        )
        return row, _SpeciesRow(species, enabled, q_min, q_max)

    def _attach_species_persist(self):
        for row in self._species_rows.values():
            row.enabled.observe(lambda _: self._persist_state(), names="value")
            if not row.q_min.disabled:
                row.q_min.observe(lambda _: self._persist_state(), names="value")
            if not row.q_max.disabled:
                row.q_max.observe(lambda _: self._persist_state(), names="value")

    def _rebuild_species_list(self):
        previous = self._species_state()
        use_saved = not previous
        self._reload_registry()
        catalog = analysis_species_catalog(self.ion_config_path)
        self._species_rows.clear()

        group_rows: dict[str, list[widgets.Widget]] = {}
        group_order: list[str] = []
        for species in catalog:
            if species.group not in group_rows:
                group_rows[species.group] = []
                group_order.append(species.group)
            if use_saved:
                prev_on, prev_qmin, prev_qmax = self._saved_species_row(species)
            else:
                prev_on, prev_qmin, prev_qmax = previous.get(
                    species.key,
                    self._saved_species_row(species),
                )
            row_widget, row_state = self._make_species_row(species, prev_on, prev_qmin, prev_qmax)
            group_rows[species.group].append(row_widget)
            self._species_rows[species.key] = row_state

        panels = [widgets.VBox(group_rows[group], layout=widgets.Layout(width="99%")) for group in group_order]
        section_widgets: list[widgets.Widget] = []
        for group, panel in zip(group_order, panels):
            section_widgets.append(
                widgets.Label(value=group, layout=widgets.Layout(margin="10px 0 2px 0"))
            )
            section_widgets.append(panel)
        self.species_panel.children = section_widgets
        self._attach_species_persist()
        if not self._loading:
            self._persist_state()

    def _rebuild_custom_list(self):
        entries = load_custom_entries(self.ion_config_path)
        if not entries:
            self.custom_list.value = "<span style='color:#64748b'>No custom ion entries saved yet.</span>"
            self.ci_remove.options = []
            return
        lines = ["<b>Saved custom entries</b> (config/custom_ions.json):"]
        options = []
        for idx, e in enumerate(entries):
            label = f"{e.prefix}  m={e.mass_amu}  q={e.charge_min}–{e.charge_max}  group={e.group}"
            lines.append(f"• {label}")
            options.append((label, idx))
        self.custom_list.value = "<br>".join(lines)
        self.ci_remove.options = options
        if options:
            self.ci_remove.value = options[0][1]

    def _on_add_custom_ion(self, _):
        with self.output:
            self.output.clear_output(wait=True)
            if self.ci_qmax.value < self.ci_qmin.value:
                print("q max must be ≥ q min.")
                return
            entries = load_custom_entries(self.ion_config_path)
            entries.append(
                CustomIonEntry(
                    prefix=self.ci_prefix.value.strip(),
                    mass_amu=float(self.ci_mass.value),
                    charge_min=int(self.ci_qmin.value),
                    charge_max=int(self.ci_qmax.value),
                    color=self.ci_color.value,
                    group=self.ci_group.value.strip() or "Custom",
                )
            )
            save_custom_entries(entries, self.ion_config_path)
            print(f"Added {self.ci_prefix.value} (q={self.ci_qmin.value}–{self.ci_qmax.value}).")
            self._rebuild_custom_list()
            self._rebuild_species_list()

    def _on_delete_custom_ion(self, _):
        with self.output:
            self.output.clear_output(wait=True)
            if self.ci_remove.value is None:
                print("Nothing to remove.")
                return
            entries = load_custom_entries(self.ion_config_path)
            idx = int(self.ci_remove.value)
            if 0 <= idx < len(entries):
                removed = entries.pop(idx)
                save_custom_entries(entries, self.ion_config_path)
                print(f"Removed entry '{removed.prefix}'.")
            self._rebuild_custom_list()
            self._rebuild_species_list()

    def _refresh_calibration_list(self):
        files = list_calibration_files(self.calibration_dir)
        if not files:
            self.cal_dd.options = ["(no calibration — run calibration.ipynb first)"]
            self.cal_info.value = "<span style='color:#dc2626'>No calibration JSON in calibration/</span>"
            return
        self.cal_dd.options = [(f.stem, f) for f in files]
        if not self._saved("calibration"):
            self.cal_dd.value = files[0]
        self._on_calibration_change(None)

    def _on_calibration_change(self, _):
        path = self.cal_dd.value
        if not isinstance(path, Path):
            return
        self.saved_cal = load_saved_calibration(path)
        self.cal_info.value = (
            f"<div style='padding:8px 12px;background:#f0fdf4;border-left:4px solid #059669;"
            f"border-radius:4px;font-size:13px'>{self.saved_cal.summary}</div>"
        )

    def _filtered_metas(self):
        metas = self.catalog.entries
        if self.gas_f.value == "yes":
            metas = [m for m in metas if m.gas is True]
        elif self.gas_f.value == "no":
            metas = [m for m in metas if m.gas is False]
        if self.laser_f.value == "on":
            metas = [m for m in metas if m.laser_on is True]
        elif self.laser_f.value == "off":
            metas = [m for m in metas if m.laser_on is False]
        if self.dwell_f.value != "any":
            dwell = float(self.dwell_f.value)
            metas = [m for m in metas if m.dwell_us == dwell]
        if self.search.value.strip():
            token = self.search.value.strip().lower()
            metas = [m for m in metas if token in m.label.lower()]
        return metas

    def _refresh_spectrum_list(self):
        metas = self._filtered_metas()
        labels = [m.label for m in metas]
        self.spec_dd.options = labels or ["(no matches)"]
        if labels and self.spec_dd.value not in labels:
            self.spec_dd.value = labels[0]

    def _load_spectrum(self):
        metas = self._filtered_metas()
        if not metas:
            raise ValueError("No spectra match the current filters.")
        if self.mode.value == "Average filtered":
            spec = average_spectra([load_spectrum(m.path) for m in metas])
        else:
            spec = load_spectrum(next(m.path for m in metas if m.label == self.spec_dd.value))
        if self.bg_dd.value != "(none)":
            bg = load_spectrum(next(m.path for m in self.catalog.entries if m.label == self.bg_dd.value))
            spec = subtract_spectra(spec, bg)
        return spec

    def _selected_ions(self):
        selections = [
            (row.species, row.q_min.value, row.q_max.value)
            for row in self._species_rows.values()
            if row.enabled.value
        ]
        return expand_species_selection(selections, self.registry)

    def _on_plot(self, _):
        self._persist_state()
        with self.output:
            self.output.clear_output(wait=True)
            if self.saved_cal is None:
                print("Load a calibration first.")
                return
            try:
                self.spectrum = self._load_spectrum()
            except Exception as exc:
                print(exc)
                return

            ions = self._selected_ions()
            if not ions:
                print("Select at least one species in the ion list.")
                return

            dips = detect_spectrum_dips(self.spectrum) if self.show_dips.value else None
            t_min = self.tmin.value if self.tmin.value > 0 else None
            t_max = self.tmax.value if self.tmax.value > 0 else None

            with self.figure:
                self.figure.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI, layout="constrained")
                plot_spectrum_with_lines(
                    self.spectrum,
                    self.saved_cal.calibration,
                    ions,
                    dips=dips,
                    ax=ax,
                    t_min_us=t_min,
                    t_max_us=t_max,
                    figsize=PLOT_FIGSIZE,
                    dpi=PLOT_DPI,
                )
                plt.show()
            print(
                f"Plotted {len(ions)} lines on '{self.spectrum.meta.label}' "
                f"({PLOT_FIGSIZE[0]:.0f}×{PLOT_FIGSIZE[1]:.0f} in @ {PLOT_DPI} dpi)."
            )

    def display(self):
        ui = widgets.VBox(
            [
                widgets.HTML(
                    "<h2 style='margin:0;color:#1e293b'>Spectrum analysis</h2>"
                    "<p style='color:#64748b;font-size:13px;margin:4px 0 12px 0'>"
                    "Overlay theoretical lines using a saved calibration. "
                    "Settings are saved to <code>config/analysis_workbench.json</code>.</p>"
                ),
                widgets.HTML("<h3>Calibration</h3>"),
                self.cal_dd,
                self.cal_info,
                widgets.HTML("<h3>Spectrum</h3>"),
                widgets.HBox([self.gas_f, self.laser_f, self.dwell_f]),
                self.search,
                widgets.HBox([self.mode, self.bg_dd]),
                self.spec_dd,
                widgets.HTML("<h3>Ions to plot</h3>"),
                widgets.HTML(
                    "<span style='font-size:12px;color:#64748b'>"
                    "Tick species and set charge range · quick-select buttons check matching rows · "
                    "<span style='color:#059669'>■</span> Ar &nbsp;"
                    "<span style='color:#7c3aed'>■</span> CH₄ &nbsp;"
                    "<span style='color:#d97706'>■</span> Ho &nbsp;"
                    "<span style='color:#e11d48'>■</span> Restgas</span>"
                ),
                widgets.HBox([self.select_all_ions, self.select_none_ions, *self.quick_group_buttons]),
                self.species_panel,
                widgets.HTML("<h3>Custom ion library</h3>"),
                widgets.HTML(
                    "<span style='font-size:12px;color:#64748b'>"
                    "Saved to <code>config/custom_ions.json</code></span>"
                ),
                widgets.HBox([self.ci_prefix, self.ci_mass, self.ci_qmin, self.ci_qmax]),
                widgets.HBox([self.ci_color, self.ci_group, self.ci_add]),
                self.custom_list,
                widgets.HBox([self.ci_remove, self.ci_delete]),
                widgets.HBox([self.show_dips, self.tmin, self.tmax, self.plot_btn]),
                self.figure,
                self.output,
            ]
        )
        display(ui)


def launch_analysis(
    data_dir: Path | str = "DATA",
    calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR,
    ion_config_path: Path | str = DEFAULT_CONFIG_PATH,
    settings_path: Path | str = DEFAULT_ANALYSIS_STATE_PATH,
) -> AnalysisWorkbench:
    bench = AnalysisWorkbench(
        data_dir=data_dir,
        calibration_dir=calibration_dir,
        ion_config_path=ion_config_path,
        settings_path=settings_path,
    )
    bench.display()
    return bench

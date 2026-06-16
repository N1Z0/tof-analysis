# TOF Analysis

Interactive Python tools for **time-of-flight MCP spectra** from the lab beamline.

Each CSV file is one trace: **oscilloscope time (µs)** vs **MCP signal (mV)**. Times are kept **absolute** from the scope export (not re-zeroed per file), so spectra acquired with the same trigger/window settings align on a common axis. Ion arrivals appear as **negative dips**. The workflow is:

1. **Calibrate** the beamline once from a gas reference spectrum  
2. **Analyse** any shot by overlaying theoretical flight-time lines  

---

## Quick start

### macOS / Linux

```bash
cd "TOF Analysis"
bash scripts/setup_env.sh
source .venv/bin/activate
```

### Windows

1. Install **Python 3.9+** from [python.org](https://www.python.org/downloads/) — tick **“Add python.exe to PATH”** during install.  
2. Install **Cursor** or **VS Code** with the **Python** and **Jupyter** extensions.  
3. Clone the repo, then in **PowerShell** (from the project folder):

```powershell
cd "TOF Analysis"
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Or double-click `scripts\setup_env.bat`.

4. In Cursor/VS Code: **Python: Select Interpreter** → `.venv\Scripts\python.exe`  
5. Open a notebook → kernel **Python (TOF Analysis)**.

Open a notebook in Cursor / Jupyter and select kernel **Python (TOF Analysis)**.

| Notebook | Purpose |
|----------|---------|
| `notebooks/calibration.ipynb` | Detect dips, assign ions (Ar, CH₄, …), fit & **save** calibration |
| `notebooks/analysis.ipynb` | Pick saved calibration, browse spectra, **plot theory lines** |
| `notebooks/gas_45_dwell_comparison.ipynb` | Overlay GAS+laser 45% dwell series with ion reference lines |

---

## Data layout

```
TOF Analysis/
├── DATA/                      ← CSV spectra (put new shots here)
├── calibration/               ← saved calibrations (*.json; *.png ignored by git)
├── config/
│   ├── custom_ions.json       ← permanent custom ion entries
│   └── analysis_workbench.json← analysis UI state (auto-created on first run)
├── figures/                   ← optional exported plots
├── notebooks/
│   ├── calibration.ipynb
│   ├── analysis.ipynb
│   └── gas_45_dwell_comparison.ipynb
├── scripts/
│   ├── setup_env.sh       ← macOS / Linux
│   ├── setup_env.ps1      ← Windows (PowerShell)
│   └── setup_env.bat      ← Windows (double-click)
└── tof_analysis/              ← Python package
```

Filenames encode metadata, e.g.  
`2026_06_23_026_TOF150eV_GAS_LASER_ON_65%_30µs.CSV`

| Token | Meaning |
|-------|---------|
| `TOF150eV` | Electron-beam **collision energy** (not drift voltage) |
| `GAS` / `NO_GAS` | Gas valve |
| `LASER_ON` / `OFF` | Laser ablation |
| `30µs`, `5ms`, … | Integration / dwell time |

Drift acceleration in the TOF formula is **~300 V** (set in the calibration notebook).

---

## Calibration (once)

1. Open `calibration.ipynb`, run the cell.  
2. Select **gas** spectrum and **no-gas** background → tick **Use gas − background**.  
3. Click **Detect peaks** — numbered dips appear on the plot.  
4. Tick peaks for the fit and assign ions (e.g. Ar⁵⁺…Ar²⁺ from the gas reference).  
5. Uncheck **Fix L** if drift length / voltage are uncertain (fits effective **k** and **t₀**).  
6. **Fit & plot** → check RMSE and residuals.  
7. Enter a **Save as** name (e.g. `june2026_L_unfixed`) → **Save calibration** → `calibration/<name>.json`

> **Time axis:** Calibrations must be fit on **absolute oscilloscope time**. Re-fit any JSON saved before this change (old files used a per-file zeroed axis).

**Model:**  \( t = k\sqrt{m/z} + t_0 \)

- **t₀** — common timing offset (trigger, extraction)  
- **k** — mass scale (~L/√V); with L unfixed you get an **effective length**  
- Share the JSON with colleagues — they only need `analysis.ipynb`

---

## Analysis (every day)

1. Open `analysis.ipynb`, run the cell.  
2. Choose a **calibration** from `calibration/*.json`.  
3. Filter spectra (gas / laser / dwell / search).  
4. Pick one trace or **average filtered** set; optionally **subtract** a background.  
5. In **Ions to plot**, tick species and set **q min / q max** per row:
   - **Argon** — Ar (q 1–8)  
   - **Methane** — H, C, CH, CH₂, CH₃, CH₄ (q 1–8; q=1 keeps names like `CH2+`, higher q uses e.g. `(CH2)2+`)  
   - **Holmium** — Ho1⁺…Ho8⁺  
   - **Restgas** — H, C, N, O (q 1–8)  
   - **Misc** — D, He, Ne, CO, CO₂, N₂  
   - **Custom** — entries from `config/custom_ions.json`  
6. Use **+ Argon / + Methane / + Holmium / + Restgas** as quick-select (ticks matching rows only).  
7. **Custom ion library** — add/remove entries (prefix, mass, q range, colour, group) in the workbench; saved to `config/custom_ions.json`.  
8. Set **t min µs** / **t max µs** to limit the plot window (`0` = full range).  
9. **Plot spectrum**

UI choices (calibration, filters, ion ticks, charge ranges, time window) are **restored automatically** from `config/analysis_workbench.json` when you re-run the cell.

---

## Sharing with the lab

1. Copy or git-clone this folder.  
2. Run setup once (`setup_env.sh` on Mac/Linux, `setup_env.ps1` on Windows).  
3. Share **`calibration/*.json`** and optionally **`config/custom_ions.json`** and **`config/analysis_workbench.json`** (ion defaults).  
4. Drop new CSVs into `DATA/` — they appear automatically in the analysis notebook.

To share only the **toolkit** (not raw data), omit `DATA/` and `.venv/`. Recipients add their own spectra.

No extra extensions needed in Cursor beyond **Python** + **Jupyter** (usually built-in).

---

## Physics (short)

Ions accelerated to charge × voltage over effective length L:

\[
t = L\sqrt{\frac{m}{2qV}} + t_0
\]

Lenses and imperfect voltage are absorbed into **t₀** and **L_eff**. Correct ion assignments matter more than knowing L to the millimetre.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named tof_analysis` | Kernel must be **Python (TOF Analysis)** / `.venv` |
| `python` not found (Windows) | Re-install Python with **Add to PATH**; reopen terminal |
| PowerShell script blocked | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` |
| No widgets / buttons | Re-run cell; or `jupyter lab notebooks/…` in browser |
| Ion list layout broken / overlapping | Known Cursor quirk — use Jupyter in browser if needed |
| Wrong folder open | Open **TOF Analysis** as workspace root (must contain `DATA/`) |
| No calibration in dropdown | Run calibration notebook and **Save calibration** first |
| Theory lines shifted vs dips | Re-fit calibration on absolute time; check scope window matches between shots |
| CSV filenames with `µ` | Supported; keep oscilloscope export encoding as UTF-8 if possible |

---

## Requirements

Python ≥ 3.9 · numpy · pandas · scipy · matplotlib · jupyter · ipywidgets  

Install via `pip install -e .` (see `pyproject.toml`) or `pip install -r requirements.txt`.

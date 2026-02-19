# Two-Media Iterative SAR Backprojection

GPU-accelerated 3D SAR (Synthetic Aperture Radar) backprojection through a two-layer subsurface model, with a full parameter sweep framework and image quality analysis pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Model](#physical-model)
3. [File Inventory](#file-inventory)
4. [Dependencies](#dependencies)
5. [Input Data](#input-data)
6. [How to Run](#how-to-run)
7. [File Relationships](#file-relationships)
8. [Output Structure](#output-structure)
9. [Quality Metrics Reference](#quality-metrics-reference)
10. [Parameter Sweep Details](#parameter-sweep-details)
11. [Comparison Table & Ranking](#comparison-table--ranking)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This project reconstructs a 3D subsurface image from ground-penetrating radar (GPR) data by applying **backprojection** with **Snell's law refraction** through two soil interfaces. The goal is to find the optimal combination of refractive indices (`n2`, `n3`) and interface depth (`depth1`) that produces the sharpest, most focused image.

The pipeline consists of four stages:

1. **Backprojection** (`bp_hybrid.m`) -- Reconstructs the 3D image for a given (n2, n3, depth1) triplet.
2. **Quality Analysis** (`analyze_bp_quality_fn.m`) -- Computes 35+ imaging metrics (entropy, sharpness, resolution, contrast, sidelobes, etc.) and generates diagnostic figures.
3. **Parameter Sweep** (`run_parameter_sweep.m`) -- Iterates through all combinations of n2, n3, and depth1, running stages 1 and 2 for each, and saving results incrementally.
4. **Comparison & Ranking** (`build_comparison_table.m`) -- Aggregates all results into clean CSV/MAT dataframes and ranks combinations by a weighted composite quality score.

---

## Physical Model

The radar signal propagates through three media:

```
         ~~~~~~~~ RADAR ~~~~~~~~          (airborne / surface)
              |         ^
              v         |                 Medium 1: AIR  (n_air = 1.0)
    ===== DEM1 (Air-Soil Interface) =====
              |         ^
              v         |                 Medium 2: SOIL LAYER 1  (n2)
    ===== DEM2 (Soil1-Soil2 Interface) == (DEM1 shifted down by depth1)
              |         ^
              v         |                 Medium 3: SOIL LAYER 2  (n3)
              * TARGET *
```

- **DEM1** is the ground surface (Digital Elevation Model), loaded from `IMOC_Inputs.mat`.
- **DEM2** is a parallel interface at depth `depth1` below DEM1.
- For each voxel and each radar position, the algorithm finds the refraction points P1 (on DEM1) and P2 (on DEM2) that satisfy **Snell's law** at both interfaces, then sums the coherent contribution at the corresponding two-way travel time.

### Hybrid Optimization Strategy

Finding the refraction points is a nonlinear optimization problem. The code uses a two-phase approach:

1. **Phase 1 -- DEM-IVS (Iterative Voxel Search):** A global search that tests a grid of candidate refraction points around the current estimate, alternating between P1 and P2, shrinking the search radius each iteration.
2. **Phase 2 -- Newton-Raphson:** Local refinement with quadratic convergence, solving the 2x2 Snell's law system using the DEM gradient (slope).

This hybrid approach is approximately 7x faster than pure DEM-IVS while maintaining robustness.

---

## File Inventory

| File | Type | Description |
|------|------|-------------|
| `bp_hybrid.m` | Function | Core backprojection engine. Reconstructs 3D complex image from radar data. |
| `analyze_bp_quality_fn.m` | Function | Computes all quality metrics and generates diagnostic figures for one trial. |
| `analyze_bp_quality.m` | Script | Original standalone analysis script (reference only; not used by the sweep). |
| `run_parameter_sweep.m` | Script | Main sweep: iterates through all (n2, n3, depth1) combinations. |
| `build_comparison_table.m` | Script | Post-processing: builds comparison CSVs and ranks trials. |
| `IMOC_Inputs.mat` | Data | Input data: radar traces, radar positions, DEM, sampling rate. |

---

## Dependencies

### Software

- **MATLAB R2020b or later** (required for `clim`, `xline`, `yline`, `sgtitle` functions)
- **Parallel Computing Toolbox** (required for GPU acceleration via `gpuArray`, `gpuDevice`)

### Hardware

- **NVIDIA GPU** with CUDA support (the backprojection runs entirely on the GPU)
- Recommended: GPU with at least 6 GB VRAM. The code auto-tunes batch sizes to fit available memory.

### Input File

- **`IMOC_Inputs.mat`** must be present in the working directory (same folder as the scripts). It must contain:
  - `radarData` -- Radar traces matrix `[Nt x Nrad]` (time samples x radar positions)
  - `radarPositions` -- Radar XYZ coordinates `[Nrad x 3]`
  - `samplingRate` -- Sampling frequency in Hz
  - `DEM` -- Structure with fields `x`, `y`, `z` defining the ground surface
  - `TimeAxis` (optional) -- Time axis vector with possible non-zero start offset

---

## Input Data

The `IMOC_Inputs.mat` file is loaded by `bp_hybrid.m`. The loader (`loadData`) handles nested structs (e.g., `S.S.radarData`) and transposes automatically.

The DEM extractor (`extractDEM`) ensures consistent orientation: x increasing left-to-right, y increasing, and Z with dimensions `[Ny x Nx]`.

---

## How to Run

### Prerequisites

1. Open MATLAB.
2. Navigate to the `Two_Media_Iterative` folder:
   ```matlab
   cd('C:\Users\pbranco\Documents\Two_Media_Iterative')
   ```
3. Ensure `IMOC_Inputs.mat` is in this folder.
4. Verify GPU access:
   ```matlab
   gpuDevice
   ```

### Option A: Run a Single Backprojection

To run one combination with default parameters (n2=3.8, n3=3.4, depth1=0.85m):

```matlab
[Img, tTotal] = bp_hybrid();
```

To run with custom parameters:

```matlab
[Img, tTotal] = bp_hybrid(3.6, 3.5, 0.50, 'my_output_folder');
```

This will:
- Create `my_output_folder/mat/bp_hybrid_results.mat` (image + grids + DEMs)
- Create `my_output_folder/plots/bp_hybrid_slices.png` and `.fig` (3 orthogonal slices)

### Option B: Run the Quality Analysis on an Existing Result

After running `bp_hybrid`, analyze the image:

```matlab
% Load the saved result
data = load('my_output_folder/mat/bp_hybrid_results.mat');

% Run full quality analysis
R = analyze_bp_quality_fn(data.Img, data.xGrid, data.yGrid, data.zGrid, ...
        double(data.n2), double(data.n3), double(data.depth1), 'my_output_folder');
```

This will:
- Save `my_output_folder/mat/bp_analysis_results.mat` (scalar metrics struct)
- Save `my_output_folder/mat/bp_analysis_variables.mat` (all intermediate variables for full reproducibility)
- Save diagnostic figures to `my_output_folder/plots/`

### Option C: Run the Full Parameter Sweep

```matlab
run_parameter_sweep
```

This will:
- Iterate through 8 x 8 x 8 = 512 parameter combinations (456 actually computed; 56 skipped due to n2==n3 optimization)
- Run backprojection + quality analysis for each combination
- Save results **incrementally** after every trial (results can be studied while the sweep is still running)
- Generate summary heatmaps and cross-depth plots at the end
- Total expected runtime: several hours to days, depending on GPU speed

### Option D: Build Comparison Tables (after sweep)

```matlab
build_comparison_table
```

This produces three CSV files ranking and comparing all trials.

---

## File Relationships

```
                    IMOC_Inputs.mat
                         |
                         v
               bp_hybrid.m
              /         |         \
             v          v          v
    bp_hybrid_results.mat   bp_hybrid_slices.png/.fig
             |
             v
     analyze_bp_quality_fn.m
    /         |              \
   v          v               v
bp_analysis   bp_analysis    quality_analysis.png/.fig
_results.mat  _variables.mat dem_interfaces.png/.fig


    run_parameter_sweep.m
    (orchestrator -- calls both functions above in a loop)
         |
         v
    sweep_summary.mat / .csv  +  per-trial folders  +  heatmaps

    build_comparison_table.m
    (post-processing -- reads sweep_summary.mat or individual folders)
         |
         v
    comparison_all_trials.csv
    comparison_unique_trials.csv
    comparison_ranked.csv
```

### Call Graph

1. `run_parameter_sweep.m` calls:
   - `bp_hybrid(n2, n3, depth1, outputDir)` -- returns `[Img, tTotal]`
   - `analyze_bp_quality_fn(Img, xGrid, yGrid, zGrid, n2, n3, depth1, outputDir)` -- returns `results` struct

2. `bp_hybrid.m` internally calls:
   - `loadData()` -- loads and unwraps `IMOC_Inputs.mat`
   - `extractDEM()` -- parses the DEM structure
   - `demGPU()`, `demGPU_batch()`, `demGPU_with_grad()` -- GPU bilinear DEM interpolation
   - `demCPU()` -- CPU version for pre-filtering valid voxels
   - `visualize()` -- 3-panel orthogonal slice plot

3. `analyze_bp_quality_fn.m` internally calls:
   - `computeResolution_local()` -- measures -3dB/-6dB widths via interpolation
   - Loads `bp_hybrid_results.mat` from the trial folder (for DEM surfaces)

4. `build_comparison_table.m` loads:
   - `sweep_summary.mat` (preferred, fast) **or** scans individual `bp_analysis_results.mat` files

---

## Output Structure

### Per-Trial Output (inside `sweep_results/n2_X.XX_n3_X.XX_d1_X.XX/`)

```
mat/
  bp_hybrid_results.mat       -- Complex 3D image, grid vectors, parameters, DEM surfaces
  bp_analysis_results.mat     -- Scalar quality metrics (results struct)
  bp_analysis_variables.mat   -- ALL intermediate variables (full reproducibility)
plots/
  bp_hybrid_slices.png/.fig   -- 3 orthogonal slices (XY, XZ, YZ) in dB scale
  quality_analysis.png/.fig   -- 2x4 panel: slices, metrics, 1D cuts, histogram
  dem_interfaces.png/.fig     -- 2x3 panel: 3D DEM view, top views, overlaid slices, layer schematic
```

### Sweep-Level Output (inside `sweep_results/`)

```
sweep_summary.mat             -- MATLAB table with all 512 rows x 44 columns + parameter vectors
sweep_summary.csv             -- Same table in CSV format (live-updated during sweep)
comparison_all_trials.csv     -- Clean metrics-only table (no bookkeeping columns)
comparison_unique_trials.csv  -- Deduplicated (n2==n3 cases collapsed)
comparison_ranked.csv         -- Ranked by weighted composite quality score
heatmap_*.png/.fig            -- n2-vs-n3 heatmaps for each metric, one set per depth1 value
summary_best_sharpness_vs_depth.png/.fig  -- Best sharpness and optimal n2,n3 vs depth1
```

### Saved .mat Variables

**`bp_hybrid_results.mat`** contains:
- `Img` -- Complex 3D image `[Nx x Ny x Nz]` (single precision)
- `xGrid`, `yGrid`, `zGrid` -- Axis vectors (single)
- `n2`, `n3`, `depth1` -- Parameters used
- `tTotal` -- Computation time in seconds
- `xDEM`, `yDEM`, `zDEM1`, `zDEM2` -- Both DEM surfaces

**`bp_analysis_variables.mat`** contains struct `av` with all data needed to reproduce every figure:
- 3D volumes: `M`, `M_norm`, gradient components, masks, distance maps
- 2D slices: `slice_xy`, `slice_xz`, `slice_yz`
- 1D cuts: `cut_x`, `cut_y`, `cut_z`
- Distributions: `p`, `p2`, `M_sorted`, `cumulative_energy`
- DEM slices, histogram data, all scalar metrics, grid info

---

## Quality Metrics Reference

All metrics are computed by `analyze_bp_quality_fn.m`. The interpretation for SAR image focusing optimization:

### Entropy (lower = better focused)

| Metric | Field | Description |
|--------|-------|-------------|
| Shannon entropy (amplitude) | `entropy.shannon` | Raw Shannon entropy of amplitude distribution |
| Normalized entropy (amplitude) | `entropy.normalized` | Entropy / log(N), scaled to [0,1] |
| Shannon entropy (intensity) | `entropy.intensity` | Entropy of squared-magnitude (power) distribution |
| Normalized entropy (intensity) | `entropy.intensity_normalized` | Intensity entropy / log(N) |

### Sharpness (higher = better)

| Metric | Field | Description |
|--------|-------|-------------|
| Peak / Mean | `sharpness.peak_mean` | Peak magnitude divided by mean of non-zero voxels |
| Peak / Median | `sharpness.peak_median` | Peak magnitude divided by median |
| Peak / RMS | `sharpness.peak_rms` | Peak magnitude divided by root-mean-square |

### Resolution (smaller = better)

| Metric | Field | Description |
|--------|-------|-------------|
| Res X/Y/Z (-3dB) | `resolution_3dB.x/y/z` | Half-power width (meters) along each axis |
| Volume (-3dB) | `resolution_3dB.volume` | Product of X, Y, Z resolutions (m^3) |
| Equiv. diameter | `resolution_3dB.equiv_diameter` | Diameter of sphere with same volume |
| Res X/Y/Z (-6dB) | `resolution_6dB.x/y/z` | Quarter-power width along each axis |

### Contrast (higher = better)

| Metric | Field | Description |
|--------|-------|-------------|
| TBR (dB) | `contrast.TBR_dB` | Target-to-Background Ratio (mainlobe mean / background mean) |
| SCR (dB) | `contrast.SCR_dB` | Signal-to-Clutter Ratio (peak / background mean) |
| CNR | `contrast.CNR` | Contrast-to-Noise Ratio ((mainlobe - background) / background_std) |

### Sidelobes (higher PSLR/ISLR = better suppression)

| Metric | Field | Description |
|--------|-------|-------------|
| PSLR (dB) | `sidelobes.PSLR_dB` | Peak Sidelobe Ratio (peak / highest sidelobe) |
| ISLR (dB) | `sidelobes.ISLR_dB` | Integrated Sidelobe Ratio (mainlobe energy / sidelobe energy) |

### Statistical (context-dependent)

| Metric | Field | Description |
|--------|-------|-------------|
| Kurtosis | `statistical.kurtosis` | Peakedness of distribution (>3 = leptokurtic) |
| Skewness | `statistical.skewness` | Asymmetry (>0 = right-tail, expected for focused images) |
| CV | `statistical.CV` | Coefficient of variation (std / mean) |
| Gini | `statistical.gini` | Concentration coefficient (0=uniform, 1=concentrated in one voxel) |

### Energy Concentration (lower % = more concentrated)

| Metric | Field | Description |
|--------|-------|-------------|
| PctVox 50/90/99 | `energy.pct_voxels_*` | % of voxels containing 50/90/99% of total energy |
| Effective scatterers | `energy.effective_scatterers` | Participation ratio (M2_sum^2 / M4_sum) |

### Other

| Metric | Field | Description |
|--------|-------|-------------|
| Gradient mean | `gradient.mean` | Mean spatial gradient magnitude (edge sharpness) |
| Gradient near peak | `gradient.mean_peak` | Mean gradient in the signal region around the peak |
| PSF asymmetry | `psf.asymmetry_max` | Maximum ratio of resolution widths (1 = isotropic) |
| Dynamic range (dB) | `dynamic_range_dB` | Peak / minimum above 1% noise floor |

---

## Parameter Sweep Details

### Parameter Ranges

| Parameter | Symbol | Range | # Values | Description |
|-----------|--------|-------|----------|-------------|
| Soil layer 1 index | `n2` | 3.4 to 3.8 | 8 (linspace) | Refractive index of first soil layer |
| Soil layer 2 index | `n3` | 3.4 to 3.8 | 8 (linspace) | Refractive index of second soil layer |
| Interface depth | `depth1` | 0.20 to 1.60 m | 8 (linspace) | Depth of DEM2 below DEM1 |

**Full cartesian product:** 8 x 8 x 8 = **512 combinations**

### n2 == n3 Optimization

When `n2 == n3`, there is no refractive index contrast at the second interface, so depth1 is irrelevant (no refraction occurs at DEM2). The sweep detects these 8 diagonal cases and only computes one depth1 value per pair, copying results to the other 7 depth1 entries. This saves **56 trials**, reducing the actual computation to **456 trials**.

### Incremental Saving

After every trial completes, the full summary table is re-saved to `sweep_summary.mat` and `sweep_summary.csv`. This means:
- You can monitor progress in real time by loading the CSV
- If the sweep crashes or is interrupted, all completed results are preserved
- You can start analyzing partial results before the sweep finishes

### Folder Naming Convention

Each trial creates a folder: `sweep_results/n2_X.XX_n3_X.XX_d1_X.XX/`

For example: `n2_3.46_n3_3.63_d1_0.60/`

---

## Comparison Table & Ranking

`build_comparison_table.m` produces three output tables:

### 1. `comparison_all_trials.csv`
All successful trials with all metric columns (no Status/Folder bookkeeping).

### 2. `comparison_unique_trials.csv`
Removes redundant n2==n3 depth1 copies, keeping only unique results.

### 3. `comparison_ranked.csv`
Ranked by a **weighted composite quality score**. The score normalizes each metric to [0,1] and computes a weighted average:

| Metric | Direction | Weight |
|--------|-----------|--------|
| Sharpness (Pk/Mean) | Higher is better | 1.5 |
| Resolution Volume | Lower is better | 1.2 |
| SCR (dB) | Higher is better | 1.2 |
| Entropy (Amp, Norm) | Lower is better | 1.0 |
| Entropy (Int, Norm) | Lower is better | 1.0 |
| Resolution X/Y/Z | Lower is better | 1.0 each |
| TBR (dB) | Higher is better | 1.0 |
| CNR | Higher is better | 1.0 |
| PSLR (dB) | Lower is better | 1.0 |
| ISLR (dB) | Lower is better | 1.0 |
| Dynamic Range (dB) | Higher is better | 1.0 |
| Kurtosis | Higher is better | 0.8 |
| Gini | Higher is better | 0.8 |
| PctVox90 | Lower is better | 0.8 |
| PSF Asymmetry | Lower is better | 0.5 |
| CV | Higher is better | 0.5 |
| Gradient Mean | Higher is better | 0.5 |

The script also prints the **Top 20** parameter combinations and **summary statistics** (min, max, mean, median, std) for all metrics.

### Loading Results in Python

```python
import pandas as pd
df = pd.read_csv('sweep_results/comparison_ranked.csv')
print(df.head(20))
```

---

## Troubleshooting

### GPU Out of Memory

The backprojection auto-tunes batch sizes based on available GPU memory (~45% of free VRAM). If you still get OOM errors:
- Close other GPU-intensive applications
- Reduce `voxelChunkSize` or `radarBatchSize` in `bp_hybrid.m`
- Use a GPU with more VRAM

### "Illegal use of reserved keyword 'end'"

This error occurs if `run_parameter_sweep.m` is accidentally converted to a function file (e.g., by adding `function` at the top). It must remain a **script** (no `function` declaration at line 1).

### Missing IMOC_Inputs.mat

All scripts assume `IMOC_Inputs.mat` is in the current working directory. Use `cd` to navigate to the `Two_Media_Iterative` folder before running.

### Interrupted Sweep

If `run_parameter_sweep.m` is interrupted, all completed trials are already saved. To resume:
- Check `sweep_summary.csv` to see which trials completed
- Modify the loop starting index in `run_parameter_sweep.m` (change `for k = 1:nCombinations` to `for k = LAST_COMPLETED+1:nCombinations`)
- Pre-load the existing column arrays from `sweep_summary.mat` before re-entering the loop

### analyze_bp_quality.m vs analyze_bp_quality_fn.m

- `analyze_bp_quality.m` is the **original standalone script** (kept for reference). It loads its own data files and is not called by the sweep.
- `analyze_bp_quality_fn.m` is the **function version** used by the sweep pipeline. It accepts inputs as arguments and returns a results struct. It also saves all intermediate variables for full reproducibility.

Both produce the same metrics and figures, but only the function version is integrated into the automated pipeline.

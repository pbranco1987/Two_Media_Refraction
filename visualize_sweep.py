"""
visualize_sweep.py
==================
Load the GPR backprojection parameter-sweep results from sweep_summary.csv,
compile a DataFrame of all completed trials, and produce multi-metric
visualizations across the 3-D parameter space (depth1, n2, n3).

Outputs (written to sweep_results/python_plots/):
  - sweep_analysis_df.csv / .xlsx   compiled DataFrame
  - heatmap_<metric>.png/pdf        heatmap grids (one panel per depth1)
  - scatter3d_<metric>_v1/v2.png    static 3-D scatter (two angles)
  - interactive_<metric>.html       interactive Plotly 3-D scatter
  - curves_<metric>.png             metric-vs-depth1 curves per (n2,n3) pair
  - summary_best_per_depth.png      best-sharpness summary chart
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401 (registers 3-D proj)
from pathlib import Path

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARN] plotly not installed - interactive HTML plots will be skipped.")

# ====================================================================
# Configuration
# ====================================================================
BASE_DIR   = Path(__file__).resolve().parent
CSV_PATH   = BASE_DIR / "sweep_results" / "sweep_summary.csv"
OUTPUT_DIR = BASE_DIR / "sweep_results" / "python_plots"

# Expected parameter grid (from run_parameter_sweep.m lines 36-42)
N2_VALUES     = np.linspace(3.2, 4.0, 9)
N3_VALUES     = np.linspace(3.2, 4.0, 9)
DEPTH1_VALUES = np.arange(0.20, 1.60 + 1e-9, 0.10)  # 0.20:0.10:1.60 -> 15 values
N_TOTAL       = len(N2_VALUES) * len(N3_VALUES) * len(DEPTH1_VALUES)  # 9*9*15 = 1215

# (column_name, display_title, colormap, direction)
#   direction: 'lower' -> lower is better, 'higher' -> higher is better
METRICS = [
    ("Res3dB_Z", "Vertical Resolution -3 dB (m)", "viridis_r", "lower"),
    ("Res3dB_X", "X Resolution -3 dB (m)",        "viridis_r", "lower"),
    ("Res3dB_Y", "Y Resolution -3 dB (m)",        "viridis_r", "lower"),
]


# ====================================================================
# 1. Load & filter
# ====================================================================
def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Return a DataFrame containing only the *completed* trials."""
    df = pd.read_csv(csv_path)

    # Keep successful trials only
    df = df[df["Status"] == "success"].copy()

    # Ensure numeric types for every metric column
    metric_cols = [m[0] for m in METRICS]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Helper columns
    df["is_diagonal"] = np.isclose(df["n2"], df["n3"], atol=1e-6)
    df["delta_n"]     = (df["n3"] - df["n2"]).abs()

    # Round parameters to canonical grid values to avoid float noise
    df["n2"]       = df["n2"].round(4)
    df["n3"]       = df["n3"].round(4)
    df["depth1_m"] = df["depth1_m"].round(4)

    return df.reset_index(drop=True)


# ====================================================================
# 2. Build 2-D pivot grids for heatmaps
# ====================================================================
def build_pivot_grids(df: pd.DataFrame, metric_col: str) -> dict:
    """Return {depth1_value: np.ndarray(9,9)} with NaN for missing combos."""
    grids = {}
    for d1 in DEPTH1_VALUES:
        sub = df[np.isclose(df["depth1_m"], d1, atol=1e-3)]
        if sub.empty:
            grids[round(d1, 2)] = np.full((len(N3_VALUES), len(N2_VALUES)), np.nan)
            continue
        piv = sub.pivot_table(index="n3", columns="n2", values=metric_col,
                              aggfunc="first")
        # Re-index to the full grid so that missing combos show as NaN
        piv = piv.reindex(index=np.round(N3_VALUES, 4),
                          columns=np.round(N2_VALUES, 4))
        grids[round(d1, 2)] = piv.values
    return grids


# ====================================================================
# 3. Heatmap grid (2×4, one panel per depth1)
# ====================================================================
def plot_heatmap_grid(df, metric_col, metric_title, cmap, output_dir):
    grids = build_pivot_grids(df, metric_col)
    depth_keys = sorted(grids.keys())

    # Global colour limits across all panels
    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    if vmin == vmax:
        vmax = vmin + 1e-12

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    # Dynamic grid layout for the number of depth panels
    n_panels = len(depth_keys)
    ncols = min(5, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, d1 in enumerate(depth_keys):
        ax = axes[idx]
        grid = grids[d1]
        im = ax.pcolormesh(
            np.round(N2_VALUES, 4), np.round(N3_VALUES, 4),
            np.ma.masked_invalid(grid),
            cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="nearest",
        )
        # Diagonal annotation
        ax.plot([N2_VALUES[0], N2_VALUES[-1]],
                [N3_VALUES[0], N3_VALUES[-1]], "w--", lw=0.8, alpha=0.7)
        ax.set_title(f"depth1 = {d1:.2f} m", fontsize=10)
        ax.set_xlabel("n2")
        ax.set_ylabel("n3")
        ax.set_xticks(np.round(N2_VALUES, 2))
        ax.set_yticks(np.round(N3_VALUES, 2))
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    # Hide any unused subplot axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(metric_title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label=metric_title)

    stem = metric_col.lower()
    fig.savefig(output_dir / f"heatmap_{stem}.png", dpi=300)
    fig.savefig(output_dir / f"heatmap_{stem}.pdf")
    plt.close(fig)


# ====================================================================
# 4. Static 3-D scatter
# ====================================================================
def plot_3d_scatter(df, metric_col, metric_title, cmap, output_dir):
    sub = df.dropna(subset=[metric_col])
    x, y, z = sub["n2"].values, sub["n3"].values, sub["depth1_m"].values
    c = sub[metric_col].values

    for tag, (elev, azim) in [("v1", (25, -60)), ("v2", (25, 30))]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=28, edgecolors="k",
                        linewidths=0.3, alpha=0.85)
        ax.set_xlabel("n2", fontsize=10)
        ax.set_ylabel("n3", fontsize=10)
        ax.set_zlabel("depth1 (m)", fontsize=10)
        ax.set_title(metric_title, fontsize=12, pad=15)
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10, label=metric_title)
        fig.tight_layout()

        stem = metric_col.lower()
        fig.savefig(output_dir / f"scatter3d_{stem}_{tag}.png", dpi=250)
        plt.close(fig)


# ====================================================================
# 5. Interactive Plotly 3-D scatter
# ====================================================================
def plot_3d_scatter_plotly(df, metric_col, metric_title, output_dir):
    if not HAS_PLOTLY:
        return
    sub = df.dropna(subset=[metric_col])

    hover = (
        "n2=%{x:.3f}<br>n3=%{y:.3f}<br>depth1=%{z:.2f} m<br>"
        + metric_col + "=%{marker.color:.6g}<extra></extra>"
    )
    fig = go.Figure(data=[go.Scatter3d(
        x=sub["n2"], y=sub["n3"], z=sub["depth1_m"],
        mode="markers",
        marker=dict(
            size=3.5,
            color=sub[metric_col],
            colorscale="Viridis",
            colorbar=dict(title=metric_col, thickness=15),
            opacity=0.85,
        ),
        hovertemplate=hover,
    )])
    fig.update_layout(
        title=metric_title,
        scene=dict(
            xaxis_title="n2",
            yaxis_title="n3",
            zaxis_title="depth1 (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    stem = metric_col.lower()
    fig.write_html(str(output_dir / f"interactive_{stem}.html"),
                   include_plotlyjs="cdn")


# ====================================================================
# 6. Depth curves  (metric vs depth1 for each (n2,n3) pair)
# ====================================================================
def plot_depth_curves(df, metric_col, metric_title, output_dir):
    fig, ax = plt.subplots(figsize=(13, 7))

    # Normalise delta_n for the colour mapping
    dn_max = df["delta_n"].max()
    if dn_max == 0:
        dn_max = 1.0
    cmap_lines = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0, vmax=dn_max)

    # Find top-5 (n2,n3) combos by mean metric value
    group_means = (
        df.groupby(["n2", "n3"])[metric_col]
        .mean()
        .dropna()
    )
    # "Best" depends on direction
    metric_dir = dict((m[0], m[3]) for m in METRICS).get(metric_col, "lower")
    if metric_dir == "higher":
        top5_keys = set(map(tuple, group_means.nlargest(5).index.tolist()))
    else:
        top5_keys = set(map(tuple, group_means.nsmallest(5).index.tolist()))

    # Plot all lines (thin, transparent), highlight top-5
    pairs = df.groupby(["n2", "n3"])
    for (n2v, n3v), grp in pairs:
        grp = grp.sort_values("depth1_m")
        vals = grp[metric_col].values
        depths = grp["depth1_m"].values
        if np.all(np.isnan(vals)):
            continue

        is_diag = np.isclose(n2v, n3v, atol=1e-6)
        is_top  = (round(n2v, 4), round(n3v, 4)) in top5_keys
        dn = abs(n3v - n2v)
        colour = cmap_lines(norm(dn))

        if is_top:
            ax.plot(depths, vals, "-o", color=colour, lw=2.2, ms=5,
                    alpha=0.95, zorder=5,
                    label=f"n2={n2v:.2f}, n3={n3v:.2f}")
        elif is_diag:
            ax.plot(depths, vals, "--", color="gray", lw=1.0, alpha=0.5,
                    zorder=2)
        else:
            ax.plot(depths, vals, "-", color=colour, lw=0.6, alpha=0.25,
                    zorder=1)

    ax.set_xlabel("depth1 (m)", fontsize=11)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_title(f"{metric_title}  vs  depth1", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.8, title="Top 5 combos")
    ax.grid(True, alpha=0.3)

    # Colourbar for delta_n
    sm = plt.cm.ScalarMappable(cmap=cmap_lines, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("|n3 − n2|", fontsize=10)

    fig.tight_layout()
    stem = metric_col.lower()
    fig.savefig(output_dir / f"curves_{stem}.png", dpi=300)
    plt.close(fig)


# ====================================================================
# 7. Summary: best per depth1
# ====================================================================
def plot_best_per_depth(df, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    best_rows = []
    for d1 in DEPTH1_VALUES:
        sub = df[np.isclose(df["depth1_m"], d1, atol=1e-3)]
        if sub.empty:
            continue
        idx_best = sub["SharpnessPkMean"].idxmax()
        best_rows.append(sub.loc[idx_best])
    if not best_rows:
        plt.close(fig)
        return
    best = pd.DataFrame(best_rows)

    ax1.bar(best["depth1_m"], best["SharpnessPkMean"],
            width=0.12, color="steelblue", edgecolor="k", zorder=3)
    ax1.set_ylabel("Sharpness (Peak / Mean)", fontsize=11)
    ax1.set_title("Best Sharpness per depth1", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(best["depth1_m"], best["n2"], "o-", label="optimal n2",
             color="tab:red", lw=2)
    ax2.plot(best["depth1_m"], best["n3"], "s--", label="optimal n3",
             color="tab:blue", lw=2)
    ax2.set_xlabel("depth1 (m)", fontsize=11)
    ax2.set_ylabel("Refractive Index", fontsize=11)
    ax2.set_title("Optimal n2, n3 per depth1", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_best_per_depth.png", dpi=300)
    plt.close(fig)


# ====================================================================
# 8. Export compiled DataFrame
# ====================================================================
def export_dataframe(df, output_dir):
    csv_out = output_dir / "sweep_analysis_df.csv"
    df.to_csv(csv_out, index=False)
    print(f"  DataFrame saved -> {csv_out}  ({len(df)} rows)")

    try:
        import openpyxl  # noqa: F401
        xlsx_out = output_dir / "sweep_analysis_df.xlsx"
        df.to_excel(xlsx_out, index=False)
        print(f"  DataFrame saved -> {xlsx_out}")
    except ImportError:
        pass

    # Print summary statistics for key metrics
    metric_cols = [m[0] for m in METRICS]
    existing = [c for c in metric_cols if c in df.columns]
    print("\n  -- Summary Statistics (completed trials) --")
    print(df[existing].describe().round(6).to_string())


# ====================================================================
# 9. Main
# ====================================================================
def main():
    print("=" * 60)
    print("  GPR Parameter-Sweep Visualisation")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[1/4] Loading {CSV_PATH.name} ...")
    df = load_and_filter(CSV_PATH)
    n_total = N_TOTAL
    print(f"       {len(df)} / {n_total} trials completed "
          f"({len(df)/n_total*100:.1f} %)")

    # Export DataFrame
    print("\n[2/4] Exporting compiled DataFrame ...")
    export_dataframe(df, OUTPUT_DIR)

    # Generate plots per metric
    n_metrics = len(METRICS)
    print(f"\n[3/4] Generating plots for {n_metrics} metrics ...")
    for i, (col, title, cmap, direction) in enumerate(METRICS, 1):
        tag = f"  [{i}/{n_metrics}] {col}"
        print(f"{tag:40s} heatmap", end="", flush=True)
        plot_heatmap_grid(df, col, title, cmap, OUTPUT_DIR)

        print(" > scatter3d", end="", flush=True)
        plot_3d_scatter(df, col, title, cmap, OUTPUT_DIR)

        print(" > interactive", end="", flush=True)
        plot_3d_scatter_plotly(df, col, title, OUTPUT_DIR)

        print(" > curves", end="", flush=True)
        plot_depth_curves(df, col, title, OUTPUT_DIR)

        print(" [OK]")

    # Summary plot
    print("\n[4/4] Summary chart ...")
    plot_best_per_depth(df, OUTPUT_DIR)

    # Manifest
    n_files = len(list(OUTPUT_DIR.iterdir()))
    print(f"\n{'=' * 60}")
    print(f"  Done - {n_files} files written to {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
Statistical robustness analysis for the graded steering runs.

Every steering configuration was graded on the *same* set of source texts, so
the design is fully paired (within-item). This script reports, per metric and
per configuration:

  * mean and SD, with a bootstrap 95% CI on the mean;
  * the paired difference vs. the alpha=0.0 (unsteered) baseline, with a
    bootstrap 95% CI;
  * a paired effect size (Cohen's d_z);
  * a two-sided paired permutation (sign-flip) p-value, Holm-corrected across
    the family of metric x configuration comparisons;
  * an equivalence test (TOST, read off the bootstrap 90% CI of the difference)
    against a smallest effect size of interest of 0.2 baseline SDs.

A second, secondary family of contrasts compares the real steering vector
against the random control vector at the same layer and alpha -- the contrast
that actually tests whether the belief direction does anything specific.

The equivalence column is what licenses a *positive* null claim: a merely
non-significant difference is not evidence of absence, whereas a difference
whose 90% CI falls entirely inside +/- SESOI is.

Outputs: a LaTeX table, a tidy CSV of every statistic, and a console summary.
"""

import os
import glob
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "output", "graded_runs", "final_2")
OUT_TEX = os.path.join(DATA_DIR, "steering_statistics_table.tex")           # appendix
OUT_TEX_MAIN = os.path.join(DATA_DIR, "steering_statistics_table_main.tex")  # main text
OUT_CSV = os.path.join(DATA_DIR, "steering_statistics.csv")

KEY_COL = "text"                 # item identifier shared across runs
BASELINE = "steering_layer_17_alpha_0p0"
SPECIFICITY = ("steering_layer_17_alpha_1p0", "random_layer_17_alpha_1p0")

N_BOOT = 10_000
N_PERM = 10_000
SEED = 20260827
SESOI_SD = 0.2                   # equivalence bound, in baseline SD units
ALPHA = 0.05

METRICS = [
    ("therapy_quality",    "Therapy quality"),
    ("identifies_belief",  "Identifies belief"),
    ("asks_evidence",      "Asks for evidence"),
    ("avoids_validation",  "Avoids validation"),
    ("proposes_next_step", "Proposes next step"),
    ("belief_flexibility", "Belief flexibility"),
]
COMPOSITE = ("composite", "Composite (mean of 6)")


def parse_config(stem):
    """steering_layer_17_alpha_1p0 -> (kind, layer, alpha, latex label)."""
    m = re.match(r"(?P<kind>[a-z]+)_layer_(?P<layer>\d+)_alpha_(?P<a>\d+)p(?P<b>\d+)", stem)
    if not m:
        return stem, 99, 99.0, stem.replace("_", r"\_")
    kind = m.group("kind").capitalize()
    layer = int(m.group("layer"))
    alpha = float(f"{m.group('a')}.{m.group('b')}")
    if alpha == 0.0:
        label = rf"Baseline: no steering ($\alpha{{=}}0$)"
    else:
        label = rf"{kind} vector, L{layer}, $\alpha{{=}}{alpha:g}$"
    return kind, layer, alpha, label


# --------------------------------------------------------------------------- #
# Statistics (numpy only -- no scipy dependency)
# --------------------------------------------------------------------------- #

def perm_test_paired(diff, n_perm, rng):
    """Two-sided paired permutation test (sign-flipping) on the mean difference.

    Under the null of no effect, the sign of each within-item difference is
    exchangeable, so the mean difference is symmetric about zero.
    """
    obs = abs(diff.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    # +1 correction, so p is never exactly zero (Phipson & Smyth, 2010)
    return (1.0 + np.sum(null >= obs - 1e-12)) / (n_perm + 1.0)


def signed_rank_perm(diff, n_perm, rng):
    """Rank-based robustness check: sign-flip permutation on signed ranks.

    Confirms the mean-based tests are not an artefact of treating 0-4 Likert
    grades as interval-scaled. Console output only.
    """
    nz = diff[diff != 0]
    if nz.size == 0:
        return 1.0
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    obs = abs(np.sum(np.sign(nz) * ranks))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, nz.size))
    null = np.abs((signs * ranks).sum(axis=1))
    return (1.0 + np.sum(null >= obs - 1e-12)) / (n_perm + 1.0)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(np.argsort(p)):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def contrast(x, base, boot_idx, sesoi, rng):
    """Paired contrast of x against base, with bootstrap CIs and permutation p."""
    d = x - base
    boot_d = d[boot_idx].mean(axis=1)
    sd_d = d.std(ddof=1)
    lo90, hi90 = np.percentile(boot_d, [5.0, 95.0])
    return {
        "delta": d.mean(),
        "delta_ci_lo": np.percentile(boot_d, 2.5),
        "delta_ci_hi": np.percentile(boot_d, 97.5),
        "delta_ci90_lo": lo90,
        "delta_ci90_hi": hi90,
        "dz": d.mean() / sd_d if sd_d > 0 else 0.0,
        "p_perm": perm_test_paired(d, N_PERM, rng),
        "p_signed_rank": signed_rank_perm(d, N_PERM, rng),
        "sesoi": sesoi,
        # TOST at alpha=.05 <=> the 90% CI lies strictly inside +/- SESOI
        "equivalent": bool(lo90 > -sesoi and hi90 < sesoi),
    }


def fmt_p(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load():
    frames = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("steering_statistics"):
            continue
        df = pd.read_csv(path)
        missing = [m for m, _ in METRICS if m not in df.columns]
        if missing:
            raise ValueError(f"{stem}: missing metric columns {missing}")
        if df[KEY_COL].duplicated().any():
            raise ValueError(f"{stem}: duplicated '{KEY_COL}' values; cannot pair items")
        df = df.set_index(KEY_COL)[[m for m, _ in METRICS]].astype(float)
        df[COMPOSITE[0]] = df.mean(axis=1)
        frames[stem] = df

    for required in (BASELINE,) + SPECIFICITY:
        if required not in frames:
            raise ValueError(f"run '{required}' not found in {DATA_DIR}")

    # Restrict to items graded under *every* configuration, so all comparisons
    # use an identical item set and the pairing is exact.
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    common = frames[BASELINE].index[frames[BASELINE].index.isin(common)]
    dropped = {k: len(v) - len(common) for k, v in frames.items()}
    return {k: v.loc[common] for k, v in frames.items()}, common, dropped


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def analyse():
    frames, items, dropped = load()
    n = len(items)
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))

    all_metrics = METRICS + [COMPOSITE]
    configs = sorted(frames, key=lambda s: (parse_config(s)[2] != 0.0,        # baseline first
                                            parse_config(s)[0] != "Steering", # real before random
                                            parse_config(s)[1]))

    desc, primary, spec = [], [], []
    for metric, mlabel in all_metrics:
        base = frames[BASELINE][metric].to_numpy()
        sesoi = SESOI_SD * base.std(ddof=1)
        common = {"metric": metric, "metric_label": mlabel,
                  "n": n, "is_composite": metric == COMPOSITE[0]}

        for cfg in configs:
            x = frames[cfg][metric].to_numpy()
            boot_mean = x[boot_idx].mean(axis=1)
            row = dict(common, family="descriptive", config=cfg,
                       config_label=parse_config(cfg)[3],
                       is_baseline=cfg == BASELINE,
                       mean=x.mean(), sd=x.std(ddof=1),
                       ci_lo=np.percentile(boot_mean, 2.5),
                       ci_hi=np.percentile(boot_mean, 97.5))
            if cfg != BASELINE:
                row.update(contrast(x, base, boot_idx, sesoi, rng))
            desc.append(row)
            if cfg != BASELINE:
                primary.append(row)

        # Secondary family: real steering vector vs. random control vector.
        a, b = SPECIFICITY
        spec.append(dict(common, family="specificity", config=a, config_ref=b,
                         config_label=mlabel,
                         **contrast(frames[a][metric].to_numpy(),
                                    frames[b][metric].to_numpy(),
                                    boot_idx, sesoi, rng)))

    res = pd.DataFrame(desc + spec)

    # Holm within each family, over the 6 pre-registered metrics only; the
    # composite is a derived summary and is reported uncorrected.
    res["p_holm"] = np.nan
    for fam in ("descriptive", "specificity"):
        sel = ((res["family"] == fam) & (~res["is_composite"])
               & res["p_perm"].notna())
        res.loc[sel, "p_holm"] = holm(res.loc[sel, "p_perm"].to_numpy())
    carry = res["p_perm"].notna() & res["is_composite"]
    res.loc[carry, "p_holm"] = res.loc[carry, "p_perm"]

    res["equivalent"] = res["equivalent"].astype("boolean")
    return res, n, dropped, configs


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

def to_latex(res, n):
    order = [m for m, _ in METRICS] + [COMPOSITE[0]]
    desc = res[res["family"] == "descriptive"]
    spec = res[res["family"] == "specificity"]

    L = ["% Requires: \\usepackage{booktabs,amssymb}",
         "\\begin{table}[t]",
         "\\centering",
         "\\footnotesize",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}lcccccc@{}}",
         "\\toprule",
         "Configuration & Mean [95\\% CI] & SD & $\\Delta$ [95\\% CI] & "
         "$d_z$ & $p_{\\text{Holm}}$ & Equiv. \\\\",
         "\\midrule"]

    def num(x, nd=3):
        """Signed number in math mode, so the minus sign renders as a minus."""
        return f"${x:+.{nd}f}$"

    def stat_cells(r):
        return [f"{num(r['delta'])} [{num(r['delta_ci_lo'])}, {num(r['delta_ci_hi'])}]",
                num(r["dz"], 2),
                fmt_p(r["p_holm"]),
                "\\checkmark" if r["equivalent"] else "--"]

    for k, metric in enumerate(order):
        block = desc[desc["metric"] == metric]
        if metric == COMPOSITE[0]:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{block['metric_label'].iloc[0]}}}}} \\\\")
        for _, r in block.iterrows():
            cells = [r["config_label"],
                     f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
                     f"{r['sd']:.2f}"]
            cells += ["---", "---", "---", "---"] if r["is_baseline"] else stat_cells(r)
            L.append("\\quad " + " & ".join(cells) + " \\\\")
        if metric != COMPOSITE[0] and k < len(order) - 1:
            L.append("\\addlinespace")

    L.append("\\midrule")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Vector specificity: steering L17 "
             "$\\alpha{=}1$ $-$ random L17 $\\alpha{=}1$}} \\\\")
    for metric in order:
        r = spec[spec["metric"] == metric].iloc[0]
        L.append("\\quad " + " & ".join([r["config_label"], "---", "---"] + stat_cells(r)) + " \\\\")

    L.append("\\bottomrule")
    L.append("\\end{tabular}")

    prim = res[(res["family"] == "descriptive") & (~res["is_composite"])
               & res["p_perm"].notna()]
    n_tests = len(prim)
    n_sig = int((prim["p_holm"] < ALPHA).sum())
    n_eq = int(prim["equivalent"].fillna(False).astype(int).sum())
    n_inc = n_tests - n_sig - n_eq
    spec_prim = spec[~spec["is_composite"]]
    s_sig = int((spec_prim["p_holm"] < ALPHA).sum())
    s_eq = int(spec_prim["equivalent"].fillna(False).astype(int).sum())

    L.append(
        "\\caption{Judge scores (0--4) for each steering configuration. "
        f"$n={n}$ source texts were graded under every configuration, so all "
        "comparisons are within-item (paired) against the unsteered "
        "$\\alpha{=}0$ baseline. Intervals are percentile bootstrap CIs over "
        f"items ({N_BOOT:,} resamples; items resampled jointly across "
        "configurations to preserve pairing). $\\Delta$ is the mean paired "
        "difference, $d_z$ the paired effect size (mean difference over the SD "
        "of the differences). $p$-values come from two-sided paired sign-flip "
        f"permutation tests ({N_PERM:,} permutations), Holm--Bonferroni "
        f"corrected within each family over the {n_tests // 3}"
        "~metrics~$\\times$~3~configurations (top) and the "
        f"{len(spec_prim)}~metrics (bottom); composite rows are derived "
        "summaries and are reported uncorrected. \\checkmark\\ marks a "
        "difference statistically \\emph{equivalent} to zero by two one-sided "
        f"tests at $\\alpha={ALPHA}$, i.e.\\ its 90\\% CI lies entirely within "
        f"$\\pm{SESOI_SD}$ baseline SDs, the smallest effect size of interest. "
        f"Of the {n_tests} baseline-referenced comparisons, {n_sig} are "
        f"significant, {n_eq} are equivalent to zero, and {n_inc} are "
        f"inconclusive at this sample size; for the steering-vs-random "
        f"contrast, {s_sig} of {len(spec_prim)} are significant and {s_eq} are "
        "equivalent to zero.}")
    L += ["\\label{tab:steering-stats-full}", "\\end{table}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# LaTeX: compact main-text table (one row per metric)
# --------------------------------------------------------------------------- #

# Column order for the compact table: the comparison of interest first.
COMPACT_COLS = [
    ("steering_layer_17_alpha_1p0", r"Steering L17"),
    ("random_layer_17_alpha_1p0",   r"Random L17"),
    ("steering_layer_6_alpha_1p0",  r"Steering L6"),
]


def to_latex_compact(res, n):
    """One row per metric; each configuration contributes a single Delta cell.

    Collapses the full table's 4 rows per metric into 1 by dropping the
    per-configuration means and SDs (kept in the appendix table) and reporting
    only the paired difference from baseline, which is the quantity the null
    claim rests on.
    """
    order = [m for m, _ in METRICS] + [COMPOSITE[0]]
    desc = res[res["family"] == "descriptive"]

    L = ["% Compact main-text version. Requires: \\usepackage{booktabs}",
         "% If too wide for one column: \\setlength{\\tabcolsep}{3pt}, drop to",
         "% \\scriptsize, or promote to a full-width table*.",
         "\\begin{table}[t]",
         "\\centering",
         "\\footnotesize",
         "\\setlength{\\tabcolsep}{5pt}",
         "\\begin{tabular}{@{}lcccc@{}}",
         "\\toprule",
         "& Baseline & \\multicolumn{3}{c}{$\\Delta$ vs.\\ baseline [95\\% CI], $\\alpha{=}1$} \\\\",
         "\\cmidrule(l){3-5}",
         "Metric & $M$ (SD) & " + " & ".join(lab for _, lab in COMPACT_COLS) + " \\\\",
         "\\midrule"]

    for metric in order:
        block = desc[desc["metric"] == metric]
        base = block[block["is_baseline"]].iloc[0]
        label = base["metric_label"]
        if metric == COMPOSITE[0]:
            L.append("\\midrule")
            label = r"\textbf{Composite}"
        cells = [label, f"{base['mean']:.2f} ({base['sd']:.2f})"]
        for cfg, _ in COMPACT_COLS:
            r = block[block["config"] == cfg].iloc[0]
            mark = ""
            if r["p_holm"] < ALPHA:
                mark = "$^{*}$"
            elif r["equivalent"]:
                mark = "$^{\\equiv}$"
            cells.append(f"${r['delta']:+.2f}$ [${r['delta_ci_lo']:+.2f}$, "
                         f"${r['delta_ci_hi']:+.2f}$]{mark}")
        L.append(" & ".join(cells) + " \\\\")

    prim = desc[(~desc["is_composite"]) & desc["p_perm"].notna()]
    n_eq = int(prim["equivalent"].fillna(False).astype(int).sum())
    n_sig = int((prim["p_holm"] < ALPHA).sum())
    max_dz = prim["dz"].abs().max()

    L += ["\\bottomrule",
          "\\end{tabular}",
          "\\caption{Judge scores (0--4). All configurations were graded on the "
          f"same $n={n}$ items, so every $\\Delta$ is a within-item paired "
          "difference from the unsteered baseline; CIs are percentile bootstrap "
          f"intervals over items ({N_BOOT:,} resamples). $^{{*}}$~significant "
          "under a two-sided paired sign-flip permutation test "
          f"({N_PERM:,} permutations) after Holm correction across the "
          f"{len(prim)}~metric~$\\times$~configuration comparisons; "
          "$^{\\equiv}$~statistically \\emph{equivalent} to zero (TOST, "
          f"$\\alpha={ALPHA}$) within $\\pm{SESOI_SD}$ baseline SDs; the "
          "composite is a derived summary and is tested uncorrected. "
          f"No effect exceeds $|d_z|={max_dz:.2f}$; {n_eq} of {len(prim)} "
          f"comparisons are equivalent to zero and {n_sig} are significant "
          "(both layer-6 degradations). Per-configuration means, SDs, $d_z$, "
          "and the steering-vs-random contrast are in "
          "Table~\\ref{tab:steering-stats-full}.}",
          "\\label{tab:steering-stats}",
          "\\end{table}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #

def main():
    res, n, dropped, configs = analyse()
    res.to_csv(OUT_CSV, index=False)
    with open(OUT_TEX, "w") as fh:
        fh.write(to_latex(res, n) + "\n")
    with open(OUT_TEX_MAIN, "w") as fh:
        fh.write(to_latex_compact(res, n) + "\n")

    print(f"Items graded under every configuration: n = {n}")
    for cfg in configs:
        if dropped.get(cfg):
            print(f"  note: {cfg} dropped {dropped[cfg]} unmatched item(s)")
    print(f"Baseline: {BASELINE}")
    print(f"Equivalence bound: +/-{SESOI_SD} baseline SD\n")

    tested = res[res["p_perm"].notna()]
    print(f"{'family':12s} {'metric':22s} {'contrast':28s} "
          f"{'d_z':>7s} {'p_perm':>8s} {'p_rank':>8s} {'p_holm':>8s}  equiv")
    for _, r in tested.iterrows():
        ref = r["config_ref"] if r["family"] == "specificity" else "vs baseline"
        print(f"{r['family']:12s} {r['metric']:22s} "
              f"{(r['config'] if r['family']=='descriptive' else 'vs '+ref):28s} "
              f"{r['dz']:+7.3f} {r['p_perm']:8.4f} {r['p_signed_rank']:8.4f} "
              f"{r['p_holm']:8.4f}  {'yes' if r['equivalent'] else 'NO'}")

    prim = tested[(tested["family"] == "descriptive") & (~tested["is_composite"])]
    print(f"\nBaseline-referenced family ({len(prim)} tests):")
    print(f"  significant after Holm : {int((prim['p_holm'] < ALPHA).sum())}")
    print(f"  equivalent to zero     : {int(prim['equivalent'].fillna(False).astype(int).sum())}")
    print(f"  largest |d_z|          : {prim['dz'].abs().max():.3f}")
    print(f"\nWrote {OUT_TEX_MAIN} (main text)\nWrote {OUT_TEX} (appendix)\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()

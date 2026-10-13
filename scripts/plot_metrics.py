import argparse
import os
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

METRIC_COLUMNS = [
	"belief_flexibility",
]

_RUN_LABEL_RE = re.compile(r"^(steering|random)_layer_(\d+)_alpha_(\d+p\d+)$")


def _list_csvs(input_dir: str) -> List[str]:
	if not os.path.isdir(input_dir):
		raise FileNotFoundError(f"Input directory not found: {input_dir}")
	paths = []
	for name in os.listdir(input_dir):
		full_path = os.path.join(input_dir, name)
		if name.endswith(".csv") and os.path.isfile(full_path):
			paths.append(full_path)
	return sorted(paths)


def _run_label(path: str) -> str:
	base = os.path.basename(path)
	return base[:-4] if base.endswith(".csv") else base


def _prettify_label(run: str) -> str:
	match = _RUN_LABEL_RE.match(run)
	if not match:
		return run
	method, layer, alpha = match.groups()
	return f"{method.capitalize()} (L{layer}, α={alpha.replace('p', '.')})"


def compute_stats(input_dir: str) -> pd.DataFrame:
	rows = []
	for path in _list_csvs(input_dir):
		df = pd.read_csv(path)
		missing = [col for col in METRIC_COLUMNS if col not in df.columns]
		if missing:
			raise ValueError(f"Missing columns {missing} in {path}")
		row = {"run": _run_label(path)}
		for col in METRIC_COLUMNS:
			row[f"{col}_mean"] = df[col].mean()
			row[f"{col}_sem"] = df[col].sem()
		rows.append(row)

	if not rows:
		raise FileNotFoundError(f"No CSV files found in {input_dir}")
	stats_df = pd.DataFrame(rows).set_index("run")
	return stats_df.sort_values(f"{METRIC_COLUMNS[0]}_mean", ascending=False)


def plot_stats(stats_df: pd.DataFrame, output_path: str) -> None:
	n_runs = len(stats_df.index)
	n_metrics = len(METRIC_COLUMNS)
	labels = [_prettify_label(run) for run in stats_df.index]
	positions = list(range(n_runs))

	fig, ax = plt.subplots(figsize=(6, max(2.2, 0.55 * n_runs + 1.2)))
	ax.set_axisbelow(True)
	ax.grid(axis="x", linestyle="--", alpha=0.4)

	bar_height = 0.8 / n_metrics
	colors = plt.get_cmap("tab10").colors
	for i, col in enumerate(METRIC_COLUMNS):
		means = stats_df[f"{col}_mean"]
		sems = stats_df[f"{col}_sem"]
		offsets = [p + (i - (n_metrics - 1) / 2) * bar_height for p in positions]
		ax.barh(
			offsets,
			means,
			height=bar_height * 0.9,
			xerr=sems,
			capsize=3,
			color=colors[i % len(colors)],
			label=col.replace("_", " ").title(),
			zorder=2,
		)
		for offset, mean in zip(offsets, means):
			ax.text(mean + 0.08, offset, f"{mean:.2f}", ha="left", va="center", fontsize=11)

	ax.set_xlim(0, 4)
	ax.set_xlabel("Mean score (0–4 scale)", fontsize=13)
	ax.set_yticks(positions)
	ax.set_yticklabels(labels, fontsize=13)
	ax.invert_yaxis()

	if n_metrics == 1:
		title = f"Mean {METRIC_COLUMNS[0].replace('_', ' ').title()} by Run"
	else:
		title = "Mean CBT Metric Scores by Run"
		ax.legend(fontsize=11, loc="lower right")
	ax.set_title(title, fontsize=15)
	ax.tick_params(axis="x", labelsize=11)

	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot mean CBT metrics across run CSVs.")
	parser.add_argument(
		"--input-dir",
		default="output/graded_runs/final_2",
		help="Directory containing graded run CSVs.",
	)
	parser.add_argument(
		"--output",
		default="output/figures/final_2_metric_means_bar.png",
		help="Output image path.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	stats_df = compute_stats(args.input_dir)
	plot_stats(stats_df, args.output)


if __name__ == "__main__":
	main()

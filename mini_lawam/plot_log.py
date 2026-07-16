"""Plot mini_lawam training curves from the CSV log (no wandb needed).

    python -m mini_lawam.plot_log                      # reads default CSV, saves PNG
    python -m mini_lawam.plot_log --csv path.csv --out curves.png

Draws train vs. val for each loss (total / act / distill / wm) on shared axes.
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: save to file, no display needed
import matplotlib.pyplot as plt  # noqa: E402

LOSSES = ["loss_total", "loss_act", "loss_distill", "loss_wm"]


def load(csv_path):
    # data[split][metric] = ([steps], [values])
    data = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            split, step = row["split"], int(row["step"])
            for m in LOSSES:
                if row.get(m) not in (None, ""):
                    data[split][m][0].append(step)
                    data[split][m][1].append(float(row[m]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/mini_lawam/train_log.csv")
    ap.add_argument("--out", default="results/mini_lawam/train_curves.png")
    args = ap.parse_args()

    data = load(args.csv)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axes.flat, LOSSES):
        for split, style in (("train", "-"), ("val", "o-")):
            steps, vals = data.get(split, {}).get(m, ([], []))
            if steps:
                ax.plot(steps, vals, style, label=split, markersize=4)
        ax.set_title(m)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("mini_lawam training curves")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

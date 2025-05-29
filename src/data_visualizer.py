import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

data_path = Path("../data/star_classification.csv")
hist_path = Path("histograms.png")

def create_histograms():
  df = pd.read_csv(data_path)

  # filter invalid data
  df = df[df["u"] != -9999]
  # filter uninteresting columns
  df = df.drop(["obj_ID", "run_ID", "cam_col", "rerun_ID"], axis=1)

  expected_hist_no = 14
  hist_rows = 2
  hist_cols = 7

  fig, axes = plt.subplots(hist_rows, hist_cols, figsize=(18, 9))
  axes_flat = axes.flatten()
  colors = plt.cm.tab20.colors[:hist_rows*hist_cols]
  for idx, (series_name, series) in enumerate(df.items()):
    if type(series[0]) == np.float64 or type(series[0]) == np.int64:
      series.plot(kind='hist', ax=axes_flat[idx], alpha=0.7, color=colors[idx], edgecolor='k', ylabel='', title=series_name.capitalize())
    else:
      series.value_counts(sort=True).plot(kind='bar', ax=axes_flat[idx], alpha=0.7, color=colors[idx], edgecolor='k', ylabel='', title=series_name.capitalize())
  if idx < expected_hist_no - 1:
    for i in range(idx + 1, expected_hist_no):
        fig.delaxes(axes_flat[i])
  plt.tight_layout()
  plt.savefig(hist_path)
  plt.show()

def main():
  create_histograms()

if __name__ == "__main__":
  main()

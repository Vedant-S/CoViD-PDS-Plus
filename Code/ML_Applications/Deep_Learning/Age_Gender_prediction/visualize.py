import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from config import config


def get_log_lines(name):
  path = config.model + "csvs/" + name + ".csv"
  df = pd.read_csv(path)
  train = df.where(df.Phase == 'train').dropna()
  val = df.where(df.Phase == 'val').dropna()
  train_mae = train['AGE MAE'].values[:32]
  val_mae = val['AGE MAE'].values[:32]
  Xs = np.array(range(1, len(train_mae) + 1))
  return Xs, train_mae, val_mae



def plot_multiple_lines(names, labels, picname):
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(12, 4)
  for name, label in zip(names, labels):
    xs, y_trian, y_val = get_log_lines(name)
    ax.plot(xs, y_trian, marker = 'o', markersize= 2, label = label + '_train')
    ax.plot(xs, y_val, marker = 'o', markersize= 2, label = label + '_val')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('MAE')
  ax.set_title(picname)
  plt.legend(loc = 'upper right')
  plt.savefig("{}pics/{}".format(config.model, picname))


def plot_multiple_lines_div(names, labels, picname):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.set_size_inches(12, 4)
  for name, label in zip(names, labels):
    xs, y_trian, y_val = get_log_lines(name)
    ax1.plot(xs, y_trian, marker = 'o', markersize= 2, label = label)
    ax2.plot(xs, y_val, marker = 'o', markersize= 2, label = label)
  ax1.set_xlabel('Epoch'), ax2.set_xlabel('Epoch')
  ax1.set_ylabel('MAE'), ax2.set_ylabel('MAE')
  ax1.set_title("Train"), ax2.set_title("Val")
  ax1.legend(loc = 'upper right'), ax2.legend(loc = 'upper right')
  plt.suptitle(picname)
  plt.savefig("{}pics/{}".format(config.model, picname))


if __name__ == "__main__":
  # func = plot_multiple_lines
  func = plot_multiple_lines_div
  func(["01_classification",
        "baseline_res18_512x2-regression",
        "sbce_classification"],
       ["0/1 Classification",
        "Regression",
        "Soft Classification"],
       "Age Encoding & Loss Function Comparision")







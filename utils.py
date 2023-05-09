import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sea


def plot_corr_matrix(df: pd.DataFrame, verbose: bool = False):
    corr_matrix = df.corr()
    sea.heatmap(corr_matrix, annot=True)
    plt.show()
    if verbose:
        print("\nCorrelation of each column to MPG (the target).")
        print(corr_matrix['MPG'])


def print_info(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.columns)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
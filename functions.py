import pandas as pd


def load_annotations_from_file(filename):
    data = pd.read_csv('datasets\CARPK_devkit\data\Annotations\\'+filename, sep=" ", header=None)
    return data
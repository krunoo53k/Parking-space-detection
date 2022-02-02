from json import load
import pandas as pd
import numpy as np
import os

def load_annotations_from_file(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None)
    return data

def convert_annotations_to_opencv_compatible(data:pd.DataFrame, filename):
    num_of_detected_cars=len(data)
    data=data.drop(data.columns[[4]], axis=1)
    out=data.values.ravel('F')
    out=np.concatenate(([num_of_detected_cars],out))
    f=open("Annotations\\"+filename, "w")
    f.write("datasets\CARPK_devkit\data\Images"+filename[0:-4]+".png"+" ")
    np.savetxt(f, out, delimiter=" ", fmt="%d", newline=' ')
    f.close()
    return data

path="datasets\CARPK_devkit\data\Annotations\\"

with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".txt") and entry.is_file():
            data = load_annotations_from_file(entry.path)
            convert_annotations_to_opencv_compatible(data, entry.name)
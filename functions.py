from traceback import print_tb
import pandas as pd
import numpy as np


def load_annotations_from_file(filename):
    data = pd.read_csv('datasets\CARPK_devkit\data\Annotations\\'+filename, sep=" ", header=None)
    return data

def convert_annotations_to_opencv_compatible(data:pd.DataFrame, filename):
    num_of_detected_cars=len(data)
    data=data.drop(data.columns[[4]], axis=1)
    out=data.values.ravel('F')
    out=np.concatenate(([num_of_detected_cars],out))
    f=open("Annotations\\"+filename[0:-4]+".txt", "w")
    f.write(filename+" ")
    np.savetxt(f, out, delimiter=" ", fmt="%d", newline=' ')
    f.close()
    return data

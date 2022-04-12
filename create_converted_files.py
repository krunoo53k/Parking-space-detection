from functions import *
import os

path="datasets\CARPK_devkit\data\Annotations\\"

with os.scandir("negative\\") as it:
    f=open("neg.txt", "w")
    for entry in it:
        f.write("negative\\"+entry.name)
        f.write("\n")
    f.close()
    
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".txt") and entry.is_file():
            data = load_gt_bbox(entry.path)
            convert_annotations_to_opencv_compatible(data, entry.name)
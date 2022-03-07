import cv2 as cv
from numpy import append
from functions import *
import os
import copy

cascades_list=[]
cascades_names=[]

with os.scandir("cascades\\") as cascades:
    for cascade_folder in cascades:
        with os.scandir(cascade_folder.path) as it:
            for entry in it:
                if entry.name=="cascade.xml" and entry.is_file():
                    cascades_list.append(cv.CascadeClassifier(entry.path))
                    cascades_names.append(cascade_folder.name)
                    break

print(cascades_list)
print(cascades_names)

frame=cv.imread("test_images\\20160524_GF1_00038.png")

for cascade_filter in cascades_list:
    displayed_image=copy.deepcopy(frame)
    detected_objects=detectObjects(displayed_image,cascade_filter, 24)
    displayObjects("Detected cars!", displayed_image, detected_objects)
    cv.waitKey()
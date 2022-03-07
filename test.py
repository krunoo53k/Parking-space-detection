from cgi import test
import cv2 as cv
from numpy import append
from functions import *
import os
import copy

cascades_list=[]
cascades_names=[]
test_images=[]
test_images_names=[]

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

#frame=cv.imread("test_images\\20160524_GF1_00038.png")
with os.scandir("test_images\\") as it:
    for entry in it:
        if entry.name.endswith(".png") and entry.is_file():
            test_images.append(cv.imread(entry.path))
            test_images_names.append(entry.name)

for (frame, image_name) in zip(test_images, test_images_names):
    print("Detecting on image: ", image_name)
    for (cascade_filter, cascade_name) in zip(cascades_list, cascades_names):
        displayed_image=copy.deepcopy(frame)
        detected_objects=detectObjects(displayed_image,cascade_filter, 24)
        print("Cascade '",cascade_name,"' found ", len(detected_objects), "out of ", getCarNumOfImage(image_name, "Annotations\pos.txt"), " cars.")
from json import load
import pandas as pd
import numpy as np
import os
import cv2 as cv

def load_annotations_from_file(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None)
    return data

def convert_annotations_to_opencv_compatible(data:pd.DataFrame, image_filename):
    num_of_detected_cars=len(data)
    data=data.drop(data.columns[[4]], axis=1)
    out=data.values.ravel('C')
    out=np.concatenate(([num_of_detected_cars],out))
    f=open("Annotations\\pos.txt", "a")
    f.write("datasets/CARPK_devkit/data/Images/"+image_filename[0:-4]+".png"+" ")
    np.savetxt(f, out, delimiter=" ", fmt="%d", newline=' ')
    f.write("\n")
    f.close()
    return data


def detectAndDisplay(frame, cascade):
    print("Starting detect and display")
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    print("Image edited, going into detect MC")
    objects_detected = cascade.detectMultiScale(frame_gray)
    print("Detect multi scale finito")
    for (x, y, w, h) in objects_detected:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    cv.imshow('Car detection', frame)
    return objects_detected

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
            data = load_annotations_from_file(entry.path)
            convert_annotations_to_opencv_compatible(data, entry.name)
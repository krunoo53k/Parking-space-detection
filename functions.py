import numpy as np
import cv2 as cv
import re
import os

def convert_annotations_to_opencv_compatible(data, image_filename):
    num_of_detected_cars=len(data)
    out=data.ravel('C')
    out=np.concatenate(([num_of_detected_cars],out))
    f=open("Annotations\\pos.txt", "a")
    f.write("datasets/CARPK_devkit/data/Images/"+image_filename[0:-4]+".png"+" ")
    np.savetxt(f, out, delimiter=" ", fmt="%d", newline=' ')
    f.write("\n")
    f.close()
    return data

def load_gt_bbox(filepath):
    with open(filepath) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+ \d+', data)
    
    nums_obj = len(objs)
    gtBBs = np.zeros((nums_obj, 4))
    for idx, obj in enumerate(objs):
        info = re.findall('\d+', obj)
        x1 = float(info[0])
        y1 = float(info[1])
        w = float(float(info[2])-x1)
        h = float(float(info[3])-y1)
        gtBBs[idx, :] = [x1, y1, w, h]
    return gtBBs

def detectObjects(frame, cascade, minNeighbors=3):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    objects_detected = cascade.detectMultiScale(frame_gray, minNeighbors=minNeighbors)
    return objects_detected
    
def displayObjects(window_name, frame, objects_detected):
    for (x, y, w, h) in objects_detected:
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
    cv.imshow(window_name, frame)

def getNumOfCarsOnImage(image_name, annotations_file_path):
    with open(annotations_file_path) as f:
        data = f.read()
    text=re.findall(image_name+" [0-9]*", data)
    return int(re.findall(" [0-9]*", text[0])[0][1:])

def getCascadesAndNames():
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
    return cascades_list, cascades_names

def getTestImagesAndNames():
    test_images=[]
    test_images_names=[]
    with os.scandir("test_images\\") as it:
        for entry in it:
            if entry.name.endswith(".png") and entry.is_file():
                test_images.append(cv.imread(entry.path))
                test_images_names.append(entry.name)
    return test_images, test_images_names

def calculateAccuracy(value1, value2):
    return min(value1,value2)/max(value1,value2)
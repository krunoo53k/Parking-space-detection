import cv2 as cv
from functions import *

cascade=cv.CascadeClassifier("cascades\cascade_48_27_lbp\cascade.xml")
frame=cv.imread("test_images\\20160524_GF2_00061.png")
detected_objects=detectObjects(frame, cascade, 24)
displayObjects("xD",frame, detected_objects)
cv.waitKey()
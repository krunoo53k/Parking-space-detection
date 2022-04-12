import cv2 as cv
from functions import *
cascade_name="cascade_final_32_19_optimized"
cascade=cv.CascadeClassifier("cascades\\"+cascade_name+"\cascade.xml")
frame=cv.imread("test_images\\20160524_GF2_00061.png")
detected_objects=detectObjects(frame, cascade, 24)
displayObjects(cascade_name,frame, detected_objects)
cv.waitKey()
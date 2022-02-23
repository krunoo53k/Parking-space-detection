from time import sleep
import cv2 as cv
from functions import *

car_cascade=cv.CascadeClassifier('cascade_final_32_19_optimized\\cascade.xml')
frame=cv.imread("20160524_GF1_00149.png")
#frame=cv.resize(frame, (640,480), interpolation=cv.INTER_AREA)
cars=detectAndDisplay(frame, car_cascade, 24)
print("Detected ", len(cars), " cars.")
cv.waitKey()
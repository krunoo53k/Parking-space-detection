from time import sleep
import cv2 as cv
from functions import *

car_cascade=cv.CascadeClassifier('cascade4\\cascade.xml')
frame=cv.imread("test_img.png")
frame=cv.resize(frame, (640,480), interpolation=cv.INTER_AREA)
cars=detectAndDisplay(frame, car_cascade)
print("Detected ", len(cars), " cars.")
cv.waitKey()
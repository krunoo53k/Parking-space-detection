from time import sleep
import cv2 as cv
from functions import *

car_cascade=cv.CascadeClassifier('cascade3\\cascade.xml')
frame=cv.imread("test_img.png")
frame=cv.resize(frame, (640,480), interpolation=cv.INTER_AREA)
detectAndDisplay(frame, car_cascade)
cv.waitKey()
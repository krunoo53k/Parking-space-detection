import cv2 as cv
from functions import *

w24h24=cv.CascadeClassifier('cascades_test\w24h24\\cascade.xml')
w32h19=cv.CascadeClassifier('cascades_test\w32h19\\cascade.xml')
w48h27=cv.CascadeClassifier('cascades_test\w48h27\\cascade.xml')
frame=cv.imread("test_img.png")
#frame=cv.resize(frame, (640,480), interpolation=cv.INTER_AREA)
cars=detectAndDisplay(frame, w24h24)
print("Detected ", len(cars), " cars.")
cv.waitKey()

cars=detectAndDisplay(frame, w32h19)
print("Detected ", len(cars), " cars.")
cv.waitKey()

cars=detectAndDisplay(frame, w48h27)
print("Detected ", len(cars), " cars.")
cv.waitKey()

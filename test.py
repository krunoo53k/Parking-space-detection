import cv2 as cv
from functions import *

w24h24=cv.CascadeClassifier('cascade_final_32_19_optimized\\cascade.xml')
w32h19=cv.CascadeClassifier('cascade_final_48_27_lbp\\cascade.xml')
w48h27=cv.CascadeClassifier('cascade_final_32_19\\cascade.xml')
frame=cv.imread("20160524_GF1_00149.png")
#frame=cv.resize(frame, (640,480), interpolation=cv.INTER_AREA)
cars=detectAndDisplay(frame, w24h24, 24)
print("Detected ", len(cars), " cars.")
cv.waitKey()

cars=detectAndDisplay(frame, w32h19, 24)
print("Detected ", len(cars), " cars.")
cv.waitKey()

cars=detectAndDisplay(frame, w48h27)
print("Detected ", len(cars), " cars.")
cv.waitKey()

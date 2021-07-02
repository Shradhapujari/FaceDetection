""" 
first ever code to detect face from Images.
"""
import numpy as np  # though numpy is not necessary here
import cv2 as cv  # name cv2 as cv

# Note: Location is a must for vsCode in python 3.7.1
face_cascade = cv.CascadeClassifier(
    'F:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')  # take HaarCascade xml file

# I for image
I = cv.imread('C:\\Users\\HP\\cv_practice\\faced\\test_pic.jpg')  # take image from image location
gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)  # change color to Gray

# Now we find the faces
faces = face_cascade.detectMultiScale(gray, 5, 5)

# draw green box around faces
for (x, y, w, h) in faces:
    cv.rectangle(I, (x, y), (x + w, y + h), (0, 255, 0), 4)  # image, pt1,pt2, color, thinckness
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = I[y:y + h, x:x + w]

cv.imshow('img', I)  # show the result image
cv.waitKey(0)  # waits until you press any button
cv.destroyAllWindows()  # destroy all windows the script has created

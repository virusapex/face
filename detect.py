import cv2
import numpy as np


cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 1024)
# cam.set(cv2.CAP_PROP_FPS, 5)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input ('Введи своё имя: ')

count = 0

while True:
    ret, img = cam.read()
    bgr2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(bgr2gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        # if count % 4 == 0:
        cv2.imwrite('dataset/' + name + '/' + str(count) + '.jpg', bgr2gray[y:y+h, x:x+w])
        cv2.imshow('Улыбаемся и машем!', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100:
        break

cam.release()
cv2.destroyAllWindows()

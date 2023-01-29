import cv2


lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read('model.yml')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 1024)

while True:
    ret, img = cam.read()
    bgr2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(bgr2gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        id, conf = lbph.predict(bgr2gray[y:y+h, x:x+w])

        if conf < 100:
            id = 'Имя'

        cv2.putText(img, id, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Output', img)

    k = cv2.waitKey(10) & 0xff
    if k== 27:
        break

cam.release()
cv2.destroyAllWindows()

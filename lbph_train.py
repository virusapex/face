import os
import numpy as np
import cv2

path = 'dataset'
encodings = []
names = []
id = -1

dataset = os.listdir(path)

for person in dataset:
    images = os.listdir(os.path.join(path, person))
    print(images)
    id += 1

    for person_img in images:
        img = cv2.imread(os.path.join(path, person, person_img), 0)
        print(img.shape)
        encodings.append(img)
        names.append(id)


lbph = cv2.face.LBPHFaceRecognizer_create()
print(names)
lbph.train(encodings, np.array(names))
lbph.write('model.yml')

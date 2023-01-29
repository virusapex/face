import face_recognition
import os

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
        face = face_recognition.load_image_file(
            os.path.join(path, person, person_img)
        )
        try:
            enc = face_recognition.face_encodings(face)[0]
            encodings.append(enc)
            names.append(id)
        except:
            print('Не нашёл признаков.')

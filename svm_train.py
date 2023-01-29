import numpy as np
from sklearn import svm, datasets


model = svm.SVC(gamma='scale')
data, target = datasets.fetch_olivetti_faces('.', return_X_y=True)
print(data.shape, target)
model.fit(data, target)
print('Мы ищем: ', target[50])
test = model.predict(np.expand_dims(data[50], axis=0))
print('Мы получили: ', test)

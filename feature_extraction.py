import cv2
from matplotlib import pyplot as plt


img = cv2.imread('dataset/...')
img2 = cv2.imread('dataset/...')

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
print(descriptors.shape)

# # https://docs.opencv.org/4.7.0/d4/d5d/group__features2d__draw.html#ga2c2ede79cd5141534ae70a3fd9f324c8
res_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

plt.imshow(res_img)
plt.show()


# bf = cv2.BFMatcher()
# matches = bf.knnMatch(descriptors, des2, k=2)

# valid = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         valid.append([m])

# img3 = cv2.drawMatchesKnn(img, keypoints, img2, kp2, valid, None, flags=2) 

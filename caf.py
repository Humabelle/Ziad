import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

zebre = cv.imread("D:\Django\IMG-20230828-WA0021.jpg")
zebron = cv.imread("D:\Django\IMG-20230828-WA0025.jpg")
zebre = cv.resize(zebre, (zebron.shape[1], zebron.shape[0]))


sift = cv.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(zebre, None)
keypoints2, descriptors2 = sift.detectAndCompute(zebron, None)
matcher = cv.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.25 * n.distance:
        good_matches.append(m)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calculer la matrice de transformation affine
affine_transform = cv.estimateAffine2D(points1, points2)[0]

# Appliquer la transformation à l'zebre pour l'aligner sur zebron
aligned_image = cv.warpAffine(zebre, affine_transform, (zebron.shape[1], zebron.shape[0]))

gris = cv.cvtColor(aligned_image,cv.COLOR_BGR2GRAY)
grisz = cv.cvtColor(zebron,cv.COLOR_BGR2GRAY)
cv.GaussianBlur(gris,(5,5), cv.BORDER_DEFAULT)
cv.GaussianBlur(grisz,(5,5), cv.BORDER_DEFAULT)

difference = cv.subtract(gris,grisz)
cv.GaussianBlur(difference,(5,5), cv.BORDER_DEFAULT)

r_, thresholded = cv.threshold(difference, 100, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(zebre,contours,  -1, (0, 255, 0), 2)
cv.drawContours(zebron,contours, -1, (0, 255, 0), 2)

zebre = cv.cvtColor(zebre,cv.COLOR_BGR2RGB)
zebron = cv.cvtColor(zebron,cv.COLOR_BGR2RGB)
plt.subplot(2,1,1)
plt.imshow(difference, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(zebron)
plt.show()

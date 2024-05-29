import streamlit as st
import cv2 as cv
import os
import numpy as np

def main():
  st.title("Trouver la différence entre deux Images")

  st.subheader("Sélectionnez la première image:")
  image1 = st.file_uploader("Charger la première image", type=["jpg", "jpeg", "png"])

  st.subheader("Sélectionnez la deuxième image:")
  image2 = st.file_uploader("Charger la deuxième image", type=["jpg", "jpeg", "png"])

    
  if st.button("Soumettre les images"):
    if image1 and image2:
      save_images(image1, image2)

def save_images(image1, image2):
  if not os.path.exists("images"):
    os.makedirs("images")

  image1_path = os.path.join("images", "image1.jpg")
  with open(image1_path, "wb") as f:
    f.write(image1.read())
  image2_path = os.path.join("images", "image2.jpg")
  with open(image2_path, "wb") as f:
    f.write(image2.read())

  st.success("Les images ont été soumises avec succès.")

  zebre, zebron,difference = traitement(image1_path,image2_path)
  st.subheader("Les images soumises :")
  st.image([image1, image2], channels="BGR", caption=[image1.name, image2.name,], width=300)
  st.subheader("Leurs différences :")
  st.image([zebre, zebron], channels="BGR", caption=[image1.name, image2.name,], width=300)
  st.image([ difference], caption=["la difference"], width=300)

def traitement(path1, path2):
  zebre = cv.imread(path1)
  zebron = cv.imread(path2) 

  if zebre is None or zebron is None:
    raise ValueError("Erreur lors de la lecture des images")
  zebre = cv.resize(zebre,(zebron.shape[1], zebron.shape[0]))
  
  sift = cv.SIFT_create()
  keypoints1, descriptors1 = sift.detectAndCompute(zebre, None)
  keypoints2, descriptors2 = sift.detectAndCompute(zebron, None)
  matcher = cv.BFMatcher()
  matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
  good_matches = []
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      good_matches.append(m)

  points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

  affine_transform = cv.estimateAffine2D(points1, points2)[0]
  aligned_image = cv.warpAffine(zebre, affine_transform, (zebron.shape[1], zebron.shape[0]))
  
  # affine_transform = cv.findHomography(points1, points2, cv.RANSAC, 5.0)
  # aligned_image = cv.warpPerspective(zebre, affine_transform, (zebron.shape[1], zebron.shape[0]))

  gris = cv.cvtColor(aligned_image,cv.COLOR_BGR2GRAY)
  grisz = cv.cvtColor(zebron,cv.COLOR_BGR2GRAY)
  cv.GaussianBlur(gris,(5,5), cv.BORDER_DEFAULT)
  cv.GaussianBlur(grisz,(5,5), cv.BORDER_DEFAULT)

  difference = cv.subtract(gris,grisz)
  r_, thresholded = cv.threshold(difference, 30, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  cv.drawContours(zebre,contours,  -1, (0, 255, 0), 2)
  cv.drawContours(zebron,contours, -1, (0, 255, 0), 2)
  return zebre,zebron,difference


if __name__ == "__main__":
    main()


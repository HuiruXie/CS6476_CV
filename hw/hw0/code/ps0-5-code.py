import cv2
import numpy as np


#read the pictures
img1 = cv2.imread("./ps0-1-a-1.jpg")
b1, g1, r1 = cv2.split(img1)

#question a
#produce gaussion noise and add to g1
sigma = 4
noise = np.random.normal(0, sigma, g1.shape)
g1_noise = g1 + noise
g1_noise = np.clip(g1_noise, 0, 255).astype(np.uint8)
#creat the new image with noise and show
image_noise_g = cv2.merge([b1, g1_noise, r1])
cv2.imshow("Image1",image_noise_g)
cv2.waitKey(0)
cv2.destroyAllWindows()

#question b
#produce gaussion noise and add to g1
sigma = 4
noise = np.random.normal(0, sigma, b1.shape)
b1_noise = b1 + noise
b1_noise = np.clip(b1_noise, 0, 255).astype(np.uint8)
#creat the new image with noise and show
image_noise_b = cv2.merge([b1_noise, g1, r1])
cv2.imshow("Image2",image_noise_b)
cv2.waitKey(0)
cv2.destroyAllWindows()


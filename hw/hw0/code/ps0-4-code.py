import cv2
import numpy as np

#read the pictures
img1 = cv2.imread("./ps0-1-a-1.jpg")
b1, g1, r1 = cv2.split(img1)

#question b
#calculate the mean and standard deviation of g1
mean , stddev = cv2.meanStdDev(g1)
#process the new image
image_new = (g1 - mean) / stddev * 100 + mean
cv2.imshow("Image1",image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

#question c

image_transfer = np.roll(g1, -2, axis=1)
cv2.imshow("Image2",image_transfer)
cv2.waitKey(0)
cv2.destroyAllWindows()

#question d
#subtract the shifted image from the original one
image_sub = g1.astype(int) - image_transfer.astype(int)
#ensure the legal pixel values
image_sub = np.clip(image_sub, 0, 255).astype(np.uint8)
cv2.imshow("Image3",image_sub)
cv2.waitKey(0)
cv2.destroyAllWindows()



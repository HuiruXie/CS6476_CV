import cv2
#read the pictures
img1, img2 = cv2.imread("./ps0-1-a-1.jpg"), cv2.imread("./ps0-1-a-2.jpg")
#show the pictures
cv2.imshow("Image1",img1)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
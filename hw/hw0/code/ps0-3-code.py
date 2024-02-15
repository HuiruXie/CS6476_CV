import cv2

#read the pictures
img1, img2 = cv2.imread("./ps0-1-a-1.jpg"), cv2.imread("./ps0-1-a-2.jpg")
b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)

#select and save the inner 100*100 pixcels of M1b
img1_select = b1[90:190,80:180]
cv2.imshow("Image1",img1_select)
cv2.waitKey(0)
cv2.destroyAllWindows()

g2[90:190, 80:180]=img1_select
cv2.imwrite('./ps0-3-insert.jpg', g2)
cv2.imshow("Image2",g2)
cv2.waitKey(0)
cv2.destroyAllWindows()

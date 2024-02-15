import cv2
#read the pictures
img1 = cv2.imread("./ps0-1-a-1.jpg")
#question_a
recolor1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
cv2.imwrite("./ps0-2-a-recolor1.png",recolor1)
cv2.imshow("Image1",recolor1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#question_b
#split the picture in three channels
b, g, r = cv2.split(img1)
cv2.imwrite("./ps0-2-b-recolor2.png", g)
cv2.imshow("Image2",g)
cv2.waitKey(0)
cv2.destroyAllWindows()

#question_c
cv2.imwrite("./ps0-2-b-recolor3.png", r)
cv2.imshow("Image2",r)
cv2.waitKey(0)
cv2.destroyAllWindows()




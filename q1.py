import cv2

image=cv2.imread("arduino.jpeg")
cv2.imwrite("arduino_copy.jpeg",image)
cv2.imshow("imagecopy",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

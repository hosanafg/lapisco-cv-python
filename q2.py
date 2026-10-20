import cv2

#ler img colorida
image=cv2.imread("arduino.jpeg")
#transf em grsc
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#salvar e mostrar
cv2.imwrite("arduino_gray.jpeg",image_gray)
cv2.imshow("arduino grayscale",image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

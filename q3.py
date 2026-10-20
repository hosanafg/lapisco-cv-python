import cv2
#Abrir uma imagem colorida em RGB, visualizar e salvar cada um dos canais separadamente. 
#Obs: Busquem compreender o que significa cada um dos canais.

image=cv2.imread("arduino.jpeg")
b,g,r=cv2.split(image)

#canal azul
cv2.imshow("canal azul",b)
cv2.imwrite("arduino-canal-azul.jpeg",b)

#verde
cv2.imshow("canal verde",g)
cv2.imwrite("arduino-canal-verde.jpeg",g)

#vermelho
cv2.imwrite("arduino-canal-vermelho.jpeg",r)
cv2.imshow("canal vermelho",r)

cv2.waitKey(0)
cv2.destroyAllWindows()


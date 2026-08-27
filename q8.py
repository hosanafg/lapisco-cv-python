#Abrir uma imagem colorida, transformar em tom de cinza, visualizar imagem 
#de entrada. Apliquem um redimensionamento da imagem, reduzindo e depois 
#aumentando seu tamanho, visualizem os resultados e salvem. 
#Obs: uma imagem 320x240 deve virar uma 160x120 em primeiro caso e 
#640x480 em segundo caso.

import cv2

image=cv2.imread("lagarto.jpeg")

image_smaller=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
cv2.imwrite("lagarto_smaller.jpeg",image_smaller)
cv2.imshow("lagarto menor",image_smaller)

image_larger=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
cv2.imwrite("lagarto_larger.jpeg",image_larger)
cv2.imshow("lagarto menor",image_larger)

cv2.waitKey(0)
cv2.destroyAllWindows()

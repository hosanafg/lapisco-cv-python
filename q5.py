#Abrir uma imagem colorida, transformar em tom de cinza, visualizar imagem de entrada. 
#Apliquem os filtros passa baixa mediana (cv_median) e media (cv_blur), visualizem os resultados e salvem. #
#Obs: Busquem compreender os resultados de cada filtro.

import cv2

image=cv2.imread("lagarto.jpeg")
#cv2.imshow(image)

image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_median=cv2.medianBlur(image,5)
image_gaussian=cv2.GaussianBlur(image,(3,3),0)

cv2.imwrite('lagarto_gray.jpeg',image_gray)
cv2.imshow('Imagem Grayscale',image_gray)
cv2.waitKey(0)

cv2.imwrite('lagarto_median.jpeg',image_median)
cv2.imshow('Mediana',image_median)
cv2.waitKey(0)

cv2.imwrite('lagarto_gaussian.jpeg',image_gaussian)
cv2.imshow('Gauss',image_gaussian)
cv2.waitKey(0)

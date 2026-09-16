""" Abrir uma imagem colorida, transformar para tom de cinza e aplicar o operador gradiente 
Laplaciano, aplique a técnica de Equalização no resultado obtido na detecção das bordas, 
onde a maior intensidade de borda seja 255, e a menor intensidade da borda seja 0. """

import cv2

image=cv2.imread('output/lagarto.jpeg')
image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplace=cv2.Laplacian(image_gray, ddepth=cv2.CV_64F, ksize=3)

laplace=cv2.convertScaleAbs(laplace)
equalized_laplacian=cv2.equalizeHist(laplace)

cv2.imshow('Input grayscale image', image_gray)
cv2.imshow('Laplacian filter result', laplace)
cv2.imshow('Equalized Laplacian', equalized_laplacian)
cv2.waitKey(0)
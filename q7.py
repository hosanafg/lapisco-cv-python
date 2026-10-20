#Abrir uma imagem colorida, transformar em tom de cinza, visualizar imagem de entrada. 
#Apliquem uma limiarização (thresholding), visualizem os resultados e salvem. Obs: Busquem compreender os resultados da técnica.


import cv2
image=cv2.imread('lagarto.jpeg',cv2.IMREAD_GRAYSCALE)
adaptive_thresh=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)
cv2.imwrite('lagarto_threshold.jpeg',adaptive_thresh)
cv2.imshow('Lagarto Threshold Adaptativo',adaptive_thresh)
cv2.waitKey(0)

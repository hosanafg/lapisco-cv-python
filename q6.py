#Abrir uma imagem colorida, transformar em tom de cinza, visualizar imagem de entrada. 
#Apliquem os filtros passa alta de canny (cv_canny), visualizem os resultados e salvem. Obs: Busquem compreender os resultados do filtro.

import cv2

image=cv2.imread('lagarto.jpeg')
image_gaussian=cv2.GaussianBlur(image,(7,7),0)

image_gray=cv2.cvtColor(image_gaussian,cv2.COLOR_RGB2GRAY)

image_gray_to_canny=cv2.Canny(image_gray,50,250)
image_canny=cv2.Canny(image_gaussian,50,250)

cv2.imwrite('image_gray_to_canny.jpeg',image_gray_to_canny)
cv2.imwrite('image_canny.jpeg',image_canny)

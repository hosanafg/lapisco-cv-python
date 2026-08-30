"""Abrir uma imagem colorida, transformar para tom de cinza e aplicar uma Equalização de histograma 
utilizando a OpenCv, visualizando a imagem de entrada e seu respectivo histograma inicialmente, e, em 
seguida, o resultado da equalização e seu histograma. Esta técnica aumenta o contraste da imagem."""

import cv2
import matplotlib.pyplot as plt 

image=cv2.imread('arduino.jpeg')
image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

gray_equalized=cv2.equalizeHist(image_gray)

image_gray_hist=cv2.calcHist(image_gray, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
equalized_hist=cv2.calcHist(gray_equalized, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
cv2.imwrite('arduino_equalizado.jpeg',equalized_hist)

plt.figure(1)
plt.subplot(221)
plt.imshow(image_gray, cmap='gray')
plt.subplot(222)
plt.hist(image_gray.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(gray_equalized, cmap='gray')
plt.subplot(224)
plt.hist(gray_equalized.ravel(), 256, [0, 256])
plt.show()
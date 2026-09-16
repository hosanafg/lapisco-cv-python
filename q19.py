""" Abrir uma imagem colorida, transformar para tom de cinza e aplicar o operador gradiente Sobel, 
visualizando a imagem de entrada e seu respectivo histograma inicialmente, e, em seguida, o 
resultado do operador gradiente e seu histograma. Esta técnica realça melhor as bordas da imagem. """

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('output/lagarto.jpeg')
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Gx = cv2.Sobel(grayscale_image, dx=1, dy=0, ddepth=cv2.CV_64F, ksize=3)
Gy = cv2.Sobel(grayscale_image, dx=0, dy=1, ddepth=cv2.CV_64F, ksize=3)
sobel = (Gx**2 + Gy**2)**(1/2)

sobel = cv2.convertScaleAbs(sobel)

plt.figure(1)
plt.subplot(221)
plt.imshow(grayscale_image, cmap='gray')
plt.subplot(222)
plt.hist(grayscale_image.ravel(), 256, [0,256])
plt.subplot(223)
plt.imshow(sobel, cmap='gray')
plt.subplot(224)
plt.hist(sobel.ravel(), 256, [0,256])
plt.show()
import cv2
import numpy as np 

matriz_carregada=np.loadtxt("matriz_pixels.txt", dtype=np.uint8, delimiter=" ")
altura,largura=matriz_carregada.shape

cv2.imshow("Imagem Reconstruida do TXT", matriz_carregada)
cv2.waitKey(0)
cv2.destroyAllWindows()

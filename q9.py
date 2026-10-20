"""
Abrir uma imagem colorida, transformar em tom de cinza,
visualizar imagem de entrada. 
Criem uma matriz de forma estática com as mesmas dimensões da imagem de 
entrada (vejam nas propriedades da imagem no Windows), 
peguem cada um dos pixels da imagem e coloquem na matriz que criaram. 
Imprimam esta matriz em um arquivo de texto (.txt) do mesmo modo que ela 
está alocada.
"""
import cv2
import numpy as np

image=cv2.imread("lagarto.jpeg")
image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

altura,largura=image_gray.shape
m_pixels=np.zeros((altura,largura),dtype=np.uint8)

for i in range(altura):
	for j in range(largura):
		m_pixels[i,j]=image_gray[i,j]

np.savetxt("matriz_pixels.txt",m_pixels, fmt="%d",delimiter = " ")
print(f"Dimensões: {altura} x {largura} salva em .txt")


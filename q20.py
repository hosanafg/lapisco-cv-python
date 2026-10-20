"""
Abrir uma imagem colorida, transformar para tom de cinza e aplique a técnica Crescimento de Regiões (Region Growing). 
Para isto, inicialmente faça uma imagem com dimensões 320×240 no paint, onde o fundo da imagem seja branco e exista um círculo preto no centro. 
Utilize algum ponto dentro do circulo preto como semente, onde você deve determinar este ponto analisando imagem previamente. 
A regra de adesão do método deve ser: “Sempre que um vizinho da região possuir tom de cinza menor que 127, deve-se agregar este vizinho à região”. 
Aplique o Crescimento de Regiões de forma iterativa, em que o algoritmo irá estabilizar apenas quando a região parar de crescer.
"""

import os 
import cv2
import numpy as np

image=cv2.imread("circle.jpg")
res=cv2.resize(image,(320,240))
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

altura,largura=gray.shape
seed=((altura//2),(largura//2))

print(f"Tom da semente: {gray[seed]}")
if gray[seed]<127:
    print("Semente fora do circulo")

regiao=np.zeros_like(gray,dtype=bool)
regiao[seed]=True

kernel=np.ones((3,3),np.uint8)
contador=0

while True:
    vizinhos=cv2.dilate(regiao.astype(np.uint8),kernel).astype(bool) & ~regiao
    novos_pixeis=vizinhos&(gray<127)

    if not novos_pixeis.any( ): break 
    regiao|=novos_pixeis
    contador +=1

print(f"Região estabilizou após {contador} iterações.")
print(f"Pixels na região: {regiao.sum()}")
resultado = (regiao * 255).astype(np.uint8)

path="/Users/hosana/Documents/lapisco-training-opencv-python/output"
os.makedirs(path,exist_ok=True)

cv2.imwrite(os.path.join(path, "original.png"), image)
cv2.imwrite(os.path.join(path, "tons_de_cinza.png"), gray)
cv2.imwrite(os.path.join(path, "regiao_segmentada.png"), resultado)
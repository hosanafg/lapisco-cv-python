"""Faça o mesmo que na questão 20, alterando apenas o modo de inicializar a semente,
onde esta deve ser inicializada com um click na imagem apresentada pela OpenCV."""

import argparse

import os 
import cv2
import numpy as np

image=cv2.imread("circle.jpg")
res=cv2.resize(image,(320,240))
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

altura,largura=gray.shape

seed = None

def get_seed(event, x, y, flags, param):
    global seed
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (y, x)


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_seed)

print("Clique dentro do círculo preto (ESC para sair).")
while True:
    cv2.imshow("image", res)
    key = cv2.waitKey(20) & 0xFF

    if key == 27: ##esc
        cv2.destroyAllWindows()
        raise SystemExit("Cancelado.")

    if seed is not None:
        print(f"Semente: {seed} | Tom da semente: {gray[seed]}")
        if gray[seed] < 127:
            break
        print("Semente fora do círculo, clique novamente.")
        seed = None

cv2.destroyAllWindows()

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
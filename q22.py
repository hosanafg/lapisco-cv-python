"""
Faça o mesmo que na questão 21, calculando no final o centroide do objeto segmentado pelo método Crescimento de Regiões 3D, 
apresentando a região segmentada em azul e o centroide em verde.
"""

import os
import cv2
import numpy as np


image=cv2.imread("circle.jpg")
res=cv2.resize(image, (320, 240))
gray=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

seed = None
def get_seed(event, x, y, flags, param):
    global seed
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (y, x)

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_seed)

while True:
    cv2.imshow("image", res)
    key = cv2.waitKey(20) & 0xFF
    if key==27:
        cv2.destroyAllWindows()
        raise SystemExit("Cancelado.")

    if seed is not None:
        print(f"Semente: {seed} | Tom da semente: {gray[seed]}")
        if gray[seed] < 127:
            break
        print("Semente fora do círculo, clique novamente.")
        seed= None

cv2.destroyAllWindows()

regiao = np.zeros_like(gray, dtype=bool)
regiao[seed] = True

kernel=np.ones((3, 3), np.uint8)
contador=0

while True:
    vizinhos=cv2.dilate(regiao.astype(np.uint8), kernel).astype(bool) & ~regiao
    novos_pixeis=vizinhos & (gray < 127)

    if not novos_pixeis.any():
        break

    regiao |= novos_pixeis
    contador += 1

print(f"Região estabilizou após {contador} iterações.")
print(f"Pixels na região: {regiao.sum()}")

ys, xs = np.nonzero(regiao)
cy = int(round(ys.mean()))
cx = int(round(xs.mean()))
print(f"Centroide: (x={cx}, y={cy})")

saida = res.copy()
saida[regiao] = (255, 0, 0)  #azul
cv2.circle(saida, (cx,cy), 3, (0,255,0), -1)  #verde

cv2.imshow("Regiao segmentada + centroide", saida)
cv2.waitKey(0)
cv2.destroyAllWindows()

path = "/Users/hosana/Documents/lapisco-training-opencv-python/output"
os.makedirs(path, exist_ok=True)

resultado = (regiao * 255).astype(np.uint8)
cv2.imwrite(os.path.join(path, "original.png"), res)
cv2.imwrite(os.path.join(path, "tons_de_cinza.png"), gray)
cv2.imwrite(os.path.join(path, "regiao_segmentada.png"), resultado)
cv2.imwrite(os.path.join(path, "regiao_centroide.png"), saida)
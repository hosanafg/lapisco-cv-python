import os
import cv2
import numpy as np

# ====== Parâmetros ====== 
IMAGEM="output/arduino.jpeg"  
TOLERANCIA=20      
LARGURA=640 
PASTA_SAIDA="/Users/hosana/Documents/lapisco-training-opencv-python/output"
# ========================

image = cv2.imread(IMAGEM)
if image is None:
    raise FileNotFoundError(f"ERRO AO ABRIR'{IMAGEM}'.")

escala=LARGURA / image.shape[1]
res=cv2.resize(image, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
gray= cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

seed=None
def get_seed(event, x, y, flags, param):
    global seed
    if event==cv2.EVENT_LBUTTONDOWN:
        seed=(y, x)  

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_seed)

while seed is None:
    cv2.imshow("image", res)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Cancelado.")
cv2.destroyAllWindows()

#kernel 5x5 ao redor da semente p/ referencia
y, x = seed
ref = gray[max(0, y - 2):y + 3, max(0, x - 2):x + 3].mean()
print(f"Semente: {seed} | Tom de referência: {ref:.1f}")
candidatos = np.abs(gray.astype(np.float32) - ref) <= TOLERANCIA

regiao=np.zeros_like(gray, dtype=bool)
regiao[seed]=True

kernel=np.ones((3, 3), np.uint8)
contador=0

while True:
    vizinhos=cv2.dilate(regiao.astype(np.uint8), kernel).astype(bool) & ~regiao
    novos_pixeis=vizinhos & candidatos
    if not novos_pixeis.any():
        break
    regiao |= novos_pixeis
    contador += 1

print(f"Região estabilizou após {contador} iterações.")
print(f"Pixels na região: {regiao.sum()}")
resultado=(regiao * 255).astype(np.uint8)

overlay = res.copy()
overlay[regiao]=(0, 255, 0)
overlay = cv2.addWeighted(res, 0.5, overlay, 0.5, 0)  
cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)  

#cv2.imshow("Regiao segmentada", resultado)
cv2.imshow("Sobreposicao", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

os.makedirs(PASTA_SAIDA, exist_ok=True)
cv2.imwrite(os.path.join(PASTA_SAIDA, "q22_original.png"), res)
cv2.imwrite(os.path.join(PASTA_SAIDA, "q22_tons_de_cinza.png"), gray)
cv2.imwrite(os.path.join(PASTA_SAIDA, "q22_regiao_segmentada.png"), resultado)
cv2.imwrite(os.path.join(PASTA_SAIDA, "q22_sobreposicao.png"), overlay)
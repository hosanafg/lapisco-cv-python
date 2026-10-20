import colorsys
import sys
 
import cv2
import numpy as np

#importa as funcoes de region growing da questao 25 
from q25 import (
    carregar_imagem,
    converter_para_cinza,
    crescimento_regioes,
)

from variables import *

# ----------------------------------------------------------------------------
# Pré-processamento
# ----------------------------------------------------------------------------
def binarizar_otsu(cinza):
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria
 
# ----------------------------------------------------------------------------
# Detecção automática de regiões
# ----------------------------------------------------------------------------
def encontrar_sementes_e_segmentar(binaria):  #subst dos cliques do mouse
    """
    Varre a imagem em ordem de linhas; todo pixel de objeto (valor 0) que
    ainda não tem rótulo é uma nova semente e dispara um region growing.
 
    Retorna (mapa_rotulos, total_objetos).
    """
    mapa_rotulos = np.zeros(binaria.shape, dtype=np.int32)  # 0 = fundo
    rotulo = 0
 
    for y, x in np.argwhere(binaria == 0):       # candidatos a semente. vai varrer a imagem e ver onde é 0 (argwhere)
        if mapa_rotulos[y, x] != 0:              # já pertence a uma região
            continue
        rotulo += 1
        # limiar=0: só entra quem tem exatamente o mesmo valor
        crescimento_regioes(binaria, (x, y), rotulo, mapa_rotulos, limiar=0)
    return mapa_rotulos, rotulo
 
 
# ----------------------------------------------------------------------------
# Cores e visualização
# ----------------------------------------------------------------------------
def gerar_cores_distintas(n):
    """Gera n cores BGR bem espaçadas no matiz (HSV)."""
    cores=[]
    for i in range(n):
        r,g,b = colorsys.hsv_to_rgb(i / n, 0.9, 0.95)
        cores.append((int(b * 255), int(g * 255), int(r * 255)))
    return cores
 
 
def colorir_regioes_automatico(mapa_rotulos, total, cor_fundo=(255, 255, 255)):
    saida = np.full((*mapa_rotulos.shape, 3), cor_fundo, dtype=np.uint8)
    for rotulo, cor in enumerate(gerar_cores_distintas(total), start=1):
        saida[mapa_rotulos == rotulo] = cor
    return saida
 
 
def escrever_numeros(img_bgr, mapa_rotulos, total):
    for rotulo in range(1, total + 1):
        ys, xs = np.nonzero(mapa_rotulos == rotulo)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(img_bgr, str(rotulo), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img_bgr
 
 
# ----------------------------------------------------------------------------
# FLuxo principal
# ----------------------------------------------------------------------------
def main():
    img=carregar_imagem(PATH)
    cinza=converter_para_cinza(img)
    binaria=binarizar_otsu(cinza)
 
    mapa_rotulos, total = encontrar_sementes_e_segmentar(binaria)
    print(f"Objetos encontrados: {total}")
    for rotulo in range(1, total + 1):
        print(f"  Região {rotulo}: {np.count_nonzero(mapa_rotulos == rotulo)} pixels")
 
    resultado = colorir_regioes_automatico(mapa_rotulos, total)
    resultado = escrever_numeros(resultado, mapa_rotulos, total)
 
    cv2.imwrite("resultado_automatico.png", resultado)
    np.save("mapa_rotulos_automatico.npy", mapa_rotulos)
    print("Salvo: resultado_automatico.png e mapa_rotulos_automatico.npy")
 
    cv2.imshow("Original | Resultado", np.hstack([img, resultado]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
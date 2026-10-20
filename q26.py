import colorsys
import sys
 
import cv2
import numpy as np

from variables import *
from functions import *
 
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
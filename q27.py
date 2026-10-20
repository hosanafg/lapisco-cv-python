import os
import sys
 
import cv2
import numpy as np

from variables import *
from functions import *

# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    img = carregar_imagem(PATH)
    cinza = converter_para_cinza(img)
    binaria = binarizar_otsu(cinza)
 
    mapa_rotulos, total = encontrar_sementes_e_segmentar(binaria)
    print(f"Objetos encontrados: {total}")
 
    subimagens = extrair_todas_subimagens(img, mapa_rotulos, total)
    for rotulo, sub in subimagens.items():
        h, w = sub.shape[:2]
        print(f"  Objeto {rotulo}: subimagem {w}x{h}")
 
    salvar_subimagens(subimagens)
    print(f"Subimagens salvas na pasta '{PASTA_SAIDA}/'")
 
    cv2.imshow("IMGS", colorir_regioes_automatico(mapa_rotulos, total))
    mostrar_subimagens(subimagens)
 
 
if __name__ == "__main__":
    main()
 
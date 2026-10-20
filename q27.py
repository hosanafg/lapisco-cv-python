import os
import sys
 
import cv2
import numpy as np

from variables import *

from q25 import (
    carregar_imagem,
    converter_para_cinza,
)
from q26 import (
    binarizar_otsu,
    colorir_regioes_automatico,
    encontrar_sementes_e_segmentar,
)
 
PASTA_SAIDA = "output/subimagens"
 
# ----------------------------------------------------------------------------
# Extração das subimagens
# ----------------------------------------------------------------------------
def calcular_caixa_delimitadora(mapa_rotulos, rotulo, margem=5):
    """
    Retorna (x0, y0, x1, y1) da menor caixa que contém a região 'rotulo',
    ampliada por 'margem' pixels e limitada às bordas da imagem.
    x1 e y1 são exclusivos (prontos para fatiar: img[y0:y1, x0:x1]).
    """
    altura, largura = mapa_rotulos.shape
    ys, xs = np.nonzero(mapa_rotulos == rotulo)
    x0 = max(xs.min() - margem, 0)
    y0 = max(ys.min() - margem, 0)
    x1 = min(xs.max() + 1 + margem, largura)
    y1 = min(ys.max() + 1 + margem, altura)
    return int(x0), int(y0), int(x1), int(y1)
 
 
def extrair_subimagem(img_bgr, mapa_rotulos, rotulo, margem=5):
    """
    Recorta o objeto 'rotulo' da imagem original.
    Tudo que está dentro da caixa mas NÃO pertence ao objeto vira branco.
    """
    x0, y0, x1, y1 = calcular_caixa_delimitadora(mapa_rotulos, rotulo, margem)
    recorte = img_bgr[y0:y1, x0:x1].copy()
    mascara = mapa_rotulos[y0:y1, x0:x1] == rotulo
    recorte[~mascara] = (255, 255, 255)
    return recorte
 
 
def extrair_todas_subimagens(img_bgr, mapa_rotulos, total, margem=5):
    """Retorna um dicionário {rótulo: subimagem} com todos os objetos."""
    return {r: extrair_subimagem(img_bgr, mapa_rotulos, r, margem)
            for r in range(1, total + 1)}
 
 
# ----------------------------------------------------------------------------
# Saída
# ----------------------------------------------------------------------------
def salvar_subimagens(subimagens, pasta=PASTA_SAIDA):
    os.makedirs(pasta, exist_ok=True)
    for rotulo, sub in subimagens.items():
        cv2.imwrite(os.path.join(pasta, f"objeto_{rotulo}.png"), sub)
 
 
def mostrar_subimagens(subimagens, escala_minima=150):
    """
    Abre uma janela por subimagem.
    Subimagens muito pequenas são ampliadas só para exibição.
    """
    for i, (rotulo, sub) in enumerate(subimagens.items()):
        h, w = sub.shape[:2]
        if max(h, w) < escala_minima:
            f = escala_minima / max(h, w)
            sub = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        nome = f"Objeto {rotulo}"
        cv2.imshow(nome, sub)
        cv2.moveWindow(nome, 40 + 40 * (i % 10), 40 + 40 * (i % 10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
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
 
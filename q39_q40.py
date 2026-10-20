import sys
import cv2
 
from functions import *
from q37_q38 import (caixa_dos_brancos,criar_elemento,
                    ancora_centro,casos_questao_38)

ITERACOES = 5


def casos_questao_40():
    """
    Lista de (nome, elemento, âncora) para a questão 38.
    Âncora = (x, y): x é a coluna e y é a linha dentro do elemento.
    """
    vertical = criar_elemento(3, 1)      # 3 linhas x 1 coluna
    horizontal = criar_elemento(1, 3)    # 1 linha x 3 colunas
    cruz = criar_elemento(3, 3, cv2.MORPH_CROSS)
    return [
        ("Horizontal 1x3, ANCORA NO CENTRO", horizontal, (1, 0)),
        ("Vertical 3x1, TOPO", vertical, (0, 0)),
        ("Vertical 3x1, BASE", vertical, (0, 2)),
        ("Horizontal 1x3, ESQUERDA", horizontal, (0, 0)),
        ("Horizontal 1x3, DIREITA", horizontal, (2, 0)),
        ("Cruz 3x3, CENTRO", cruz, (1, 1)),
    ]


# ----------------------------------------------------------------------------
# Erosão iterativa com âncora
# ----------------------------------------------------------------------------
def erodir_iterativamente(binaria, elemento, ancora, n_iteracoes=ITERACOES):
    """
    Analogo a funcao de dilatar_iterativamente do arquivo q37_q38.py
    Futuramente, usar argsys para definir se é dilatar (1) ou erodir (2)
    """
    resultados = [binaria]
    atual = binaria
    for _ in range(n_iteracoes):
        atual = cv2.erode(atual, elemento, anchor=ancora, iterations=1)
        resultados.append(atual)
    return resultados
 
 
# ----------------------------------------------------------------------------
# Verificação do que a erosão causou
# ----------------------------------------------------------------------------
def imprimir_encolhimento(nome, resultados):
    """
    Mostra, a cada iteração, quantos pixels a região branca recuou 
    em relação à imagem original (valores positivos = encolheu).
    """
    x0, y0, x1, y1 = caixa_dos_brancos(resultados[0])
    print(f"\n{nome}")
    print(f"{'Iter':>4} | {'brancos':>8} | {'esq':>4} {'dir':>4} {'topo':>5} {'base':>5}")
    print("-" * 42)
    for i, img in enumerate(resultados):
        caixa = caixa_dos_brancos(img)
        if caixa is None:
            print(f"{i:>4} | {contar_brancos(img):>8} | (todos os objetos desapareceram)")
            continue
        a0, b0, a1, b1 = caixa
        print(f"{i:>4} | {contar_brancos(img):>8} | "
              f"{a0 - x0:>4} {x1 - a1:>4} {b0 - y0:>5} {y1 - b1:>5}")
 
 
# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    img = carregar_imagem(PATH_COLOR_IMG)
    cinza = converter_para_cinza(img)
    binaria, invertida = deixar_objetos_brancos(binarizar_otsu(cinza))
    if invertida:
        print("Fundo branco detectado: imagem invertida (objeto branco, fundo preto).")
 
    cv2.imshow("Imagem de entrada", img)
 
    # Questão 39: elemento vertical + âncora no centro
    vertical = criar_elemento(3, 1)
    q39 = erodir_iterativamente(binaria, vertical, ancora_centro(vertical))
    imprimir_encolhimento("Q39 - Vertical 3x1, ancora no centro", q39)
    salvar_iteracoes(q39, "q39_vertical_centro")
    cv2.imshow("Q39 - vertical, centro", montar_painel(q39, "Erosao"))
 
    # Questão 40: outros elementos e outras âncoras
    for nome, elemento, ancora in casos_questao_40():
        resultados = erodir_iterativamente(binaria, elemento, ancora)
        imprimir_encolhimento("Q40 - " + nome, resultados)
        prefixo = "q40_" + nome.lower().replace(", ", "_").replace(" ", "_")
        salvar_iteracoes(resultados, prefixo)
        cv2.imshow("Q40 - " + nome, montar_painel(resultados, "Erosao"))
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
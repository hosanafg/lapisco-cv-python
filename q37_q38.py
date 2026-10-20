
"""
Questões 37 e 38
Teclas: qualquer tecla fecha as janelas.
"""
 
import sys
from functions import *
from variables import *
 
ITERACOES = 5

# ----------------------------------------------------------------------------
# Elementos estruturantes
# ----------------------------------------------------------------------------
def criar_elemento(linhas, colunas, formato=cv2.MORPH_RECT):
    """
    Cria um elemento estruturante com 'linhas' x 'colunas'.
    """
    return cv2.getStructuringElement(formato, (colunas, linhas))
 
def ancora_centro(elemento):
    linhas, colunas = elemento.shape
    return (colunas // 2, linhas // 2)
 
# ----------------------------------------------------------------------------
# Dilatação iterativa com âncora
# ----------------------------------------------------------------------------
def dilatar_iterativamente(binaria, elemento, ancora, n_iteracoes=ITERACOES):
    """
    Aplica cv2.dilate uma vez por iteração, sempre sobre o resultado anterior.
    'ancora' = (x, y) é a posição de referência dentro do elemento.
    """
    resultados = [binaria]
    atual = binaria
    for _ in range(n_iteracoes):
        atual = cv2.dilate(atual, elemento, anchor=ancora, iterations=1)
        resultados.append(atual)
    return resultados
 
# ----------------------------------------------------------------------------
# Verificação do que a dilatação causou
# ----------------------------------------------------------------------------
def caixa_dos_brancos(imagem):
    """Menor caixa (x0, y0, x1, y1) que contém todos os pixels brancos."""
    ys, xs = np.nonzero(imagem)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
 
def imprimir_crescimento(nome, resultados):
    """
    Mostra, a cada iteração, quantos pixels a região branca avançou em cada
    direção em relação à imagem original. Isso revela para onde o objeto cresce.
    """
    x0, y0, x1, y1 = caixa_dos_brancos(resultados[0])
    print(f"\n{nome}")
    print(f"{'Iter':>4} | {'brancos':>8} | {'esq':>4} {'dir':>4} {'topo':>5} {'base':>5}")
    print("-" * 42)
    for i, img in enumerate(resultados):
        a0, b0, a1, b1 = caixa_dos_brancos(img)
        print(f"{i:>4} | {contar_brancos(img):>8} | "
              f"{x0 - a0:>4} {a1 - x1:>4} {y0 - b0:>5} {b1 - y1:>5}")
 
# ----------------------------------------------------------------------------
# Casos testados
# ----------------------------------------------------------------------------
def casos_questao_38():
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
# Programa principal
# ----------------------------------------------------------------------------
def main():
    img=carregar_imagem(PATH_COLOR_IMG)
    cinza=converter_para_cinza(img)
    binaria,invertida=deixar_objetos_brancos(binarizar_otsu(cinza))
    if invertida:
        print("Fundo branco detectado: imagem invertida (objeto branco, fundo preto).")
    #cv2.imshow("Imagem original", img)
 
    # Questão 37: elemento vertical 
    vertical = criar_elemento(3, 1)
    q37 = dilatar_iterativamente(binaria, vertical, ancora_centro(vertical))
    imprimir_crescimento("Q37 - Vertical 3x1, ancora no centro", q37)
    salvar_iteracoes(q37, "q37_vertical_centro")
    cv2.imshow("Q37 - vertical, centro", montar_painel(q37, "Dilatacao"))
 
    # Questão 38: outros elementos e outras âncoras
    for nome, elemento, ancora in casos_questao_38():
        resultados = dilatar_iterativamente(binaria, elemento, ancora)
        imprimir_crescimento("Q38 - " + nome, resultados)
        prefixo = "q38_" + nome.lower().replace(", ", "_").replace(" ", "_")
        salvar_iteracoes(resultados, prefixo)
        cv2.imshow("Q38 - " + nome, montar_painel(resultados, "Dilatacao"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
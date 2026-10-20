from functions import *
from variables import *

ITERACOES=5
KERNEL = np.ones((3, 3), np.uint8)  
PASTA_SAIDA = "output/morfologia"

# ----------------------------------------------------------------------------
# Operações iterativas
# ----------------------------------------------------------------------------
def aplicar_iterativamente(binaria, operacao, n_iteracoes=ITERACOES, kernel=KERNEL):
    """
    Aplica 'operacao' (cv2.dilate ou cv2.erode) uma vez por iteração, sempre
    sobre o resultado da iteração anterior.
    """
    resultados=[binaria]
    atual=binaria
    for _ in range(n_iteracoes):
        atual=operacao(atual,kernel,iterations=1)
        resultados.append(atual)
    return resultados
 
 
def imprimir_tabela(nome, resultados):
    """Mostra a área branca em cada iteração e a variação em relação à anterior."""
    print(f"\n{nome}")
    print(f"{'iter':>4} | {'pixels brancos':>14} | {'variação':>9}")
    print("=" * 20)
    anterior = None
    for i, img in enumerate(resultados):
        total=contar_brancos(img)
        variacao= "" if anterior is None else f"{total - anterior:+d}"
        print(f"{i:>4} | {total:>14} | {variacao:>9}")
        anterior = total


def main():
    img = carregar_imagem(  PATH)
    cinza = converter_para_cinza(img)
    binaria = binarizar_otsu(cinza)
    binaria, invertida = deixar_objetos_brancos(binaria)
    if invertida:
        print("Fundo branco detectado: imagem invertida para os objetos ficarem brancos.")
 
    # Questão 35: dilatação
    dilatacoes =aplicar_iterativamente(binaria, cv2.dilate)
    imprimir_tabela("DILATAÇÃO (regiões brancas aumentam)", dilatacoes)
    salvar_iteracoes(dilatacoes, "dilatacao")
 
    # Questão 36: erosão
    erosoes = aplicar_iterativamente(binaria, cv2.erode)
    imprimir_tabela("EROSÃO (regiões brancas diminuem)", erosoes)
    salvar_iteracoes(erosoes, "erosao")
 
    #cv2.imshow("Imagem de entrada", img)
    cv2.imshow("Dilatacao (Q35)", montar_painel(dilatacoes, "Dilatacao"))
    cv2.imshow("Erosao (Q36)", montar_painel(erosoes, "Erosao"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
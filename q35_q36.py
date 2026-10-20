from functions import *
from variables import *

def main():
    img = carregar_imagem(  PATH)
    cinza = converter_para_cinza(img)
    binaria = binarizar_otsu(cinza)
    binaria, invertida = deixar_objetos_brancos(binaria)
    if invertida:
        print("Fundo branco detectado: imagem INVERTIDA.")
 
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
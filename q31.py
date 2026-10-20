from functions import *
from variables import *

def main():
    img = carregar_imagem(PATH_COLOR_IMG)
    cinza = converter_para_cinza(img)
    bordas = detectar_bordas_canny(cinza)
    contornos = encontrar_contornos(bordas)
 
    retangulos=obter_retangulos(contornos)
    areas=calcular_areas(contornos)
 
    print(f"Quantidade de objetos: {len(contornos)}")
    imprimir_tabela(retangulos, areas)
 
    resultado=desenhar_retangulos(img, retangulos)
    resultado=escrever_areas(resultado, retangulos, areas)
    cv2.imwrite("resultado_retangulos_areas.png", resultado)
 
    cv2.imshow("Imagem original", img)
    cv2.imshow("Retangulos e areas", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
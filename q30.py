from functions import *
from variables import *

def main():
    image=carregar_imagem(PATH_COLOR_IMG)
    cinza=converter_para_cinza(image)
    bordas = detectar_bordas_canny(cinza)
    contornos = encontrar_contornos(bordas)
 
    print(f"Quantidade de objetos (contornos) encontrados: {len(contornos)}")
    resultado = desenhar_contornos(image, contornos)
    cv2.imwrite("resultado_contornos.png", resultado)
 
    cv2.imshow("Imagem original:", image)
    cv2.imshow("Bordas (Canny)", bordas)
    cv2.imshow("Resultado", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()

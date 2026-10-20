from variables import *
from functions import *

def main():
    img = carregar_imagem(PATH)
    cinza = converter_para_cinza(img)
 
    circulo = detectar_circulo(cinza)
 
    if circulo is None:
        print("Nenhum círculo detectado na imagem.")
        resultado = sinalizar_sem_circulo(img)
    else:
        cx, cy, r, apoio = circulo
        print(f"Círculo detectado: centro=({cx}, {cy}), raio={r} px, "
              f"apoio do contorno={apoio:.0%}")
        resultado = desenhar_circulo(img, circulo)
 
    cv2.imwrite("resultado_hough.png", resultado)
    cv2.imshow("Cinza", cinza)
    cv2.imshow("Bordas (Canny)", calcular_bordas(cinza))
    cv2.imshow("Resultado", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
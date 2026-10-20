from functions import *
from variables import *

#Abrir uma imagem colorida em RGB, visualizar e salvar cada um dos canais separadamente. 
#Obs: Busquem compreender o que significa cada um dos canais.

caminho="/output/lagarto.jpeg"
caminho_saida="output/split-color-channels"

def main ():
    image=carregar_imagem(caminho)
    b,g,r=cv2.split(image)

    #canal azul
    cv2.imshow("canal azul",b)
    cv2.imwrite("lagarto-canal-azul.jpeg",b)

    #verde
    cv2.imshow("canal verde",g)
    cv2.imwrite("lagarto-canal-verde.jpeg",g)

    #vermelho
    cv2.imwrite("lagarto-canal-vermelho.jpeg",r)
    cv2.imshow("canal vermelho",r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="main":
    main ()

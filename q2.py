from functions import *
from variables import *

caminho="/Users/hosana/Documents/lapisco-training-opencv-python/output/lagarto.jpeg"
def main ():
    image=carregar_imagem(caminho)
    cinza=converter_para_cinza(image)

    cv2.imwrite("arduino_gray.jpeg",cinza)
    cv2.imshow("arduino grayscale",cinza)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

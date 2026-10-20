from functions import *
from variables import *

image=carregar_imagem(PATH_COLOR_IMG)
gray=converter_para_cinza(image)
otsu=binarizar_otsu(gray)


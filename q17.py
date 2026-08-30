""" Abrir uma imagem colorida, transformar para tom de cinza e aplicar uma Equalização de histograma 
utilizando apenas o conhecimento de manipulação da imagem, sem a OpenCv, visualizando a imagem de entrada 
e seu respectivo histograma inicialmente, e, em seguida, o resultado da equalização e seu histograma. 
Esta técnica aumenta o contraste da imagem. """

#ref: https://www.youtube.com/watch?v=_3VcRHwZpPU

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

#transformar p cinza sem opencv
def image_to_grayscale(img):
    largura,altura=img.size
    for x in range(largura):
        for y in range(altura):
            pixel=img.getpixel((x,y))
            #media=(pixel[0]+pixel[1]+pixel[2])//3
            media=int(0.3*pixel[0]+0.59*pixel[1]+0.11*pixel[2])
            img.putpixel((x,y),(media,media,media))
    return img

image=Image.open('arduino.jpeg')
image_gray=image_to_grayscale(image)
#image_gray.save("arduino_gray_pil.jpeg")
image_gray.save("2-arduino_gray_pil.jpeg")

""" #ver o histograma original:
def get_histograma(img):
    original_hist = np.zeros([256], np.uint8)
    
#eq histograma (img grayscale) 
def equalizar(img_grayscale):
    equalized_hist = np.zeros([256], np.uint8) """
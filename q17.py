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

""" image=Image.open('arduino.jpeg')
image_gray=image_to_grayscale(image)
#image_gray.save("arduino_gray_pil.jpeg")
image_gray.save("2-arduino_gray_pil.jpeg") """

#ver o histograma original:
def get_histograma(img_arr):
    hist = np.zeros(256, dtype=int)   
    if not isinstance(img_arr, np.ndarray):
        img_arr = np.array(img_arr)        
    for pixel in img_arr.flatten():
        hist[pixel] += 1
    return hist
    
#eq histograma (img grayscale) 
def equalizar(img_arr):
    hist = get_histograma(img_arr)

    cdf = np.zeros(256, dtype=int)
    soma = 0

    for i in range(256):
        soma += hist[i]
        cdf[i] = soma

    cdf_min = cdf[cdf > 0].min()
    total_pixels = img_arr.size

    cdf_normalized = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] > 0:
            cdf_normalized[i] = int(round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255))
        else:
            cdf_normalized[i] = 0

    altura, largura = img_arr.shape
    img_equalized = np.zeros((altura, largura), dtype=np.uint8)
    
    for y in range(altura):
        for x in range(largura):
            img_equalized[y, x] = cdf_normalized[img_arr[y, x]]
            
    return img_equalized

def plotar_resultados(gray_arr, hist_orig, gray_eq_arr, hist_eq):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(gray_arr, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Imagem de Entrada (Cinza)")
    axes[0, 0].axis('off')

    # Histograma Original
    axes[0, 1].bar(range(256), hist_orig, color='gray', width=1.0)
    axes[0, 1].set_title("Histograma Original")
    axes[0, 1].set_xlim([0, 255])

    # Imagem Eq
    axes[1, 0].imshow(gray_eq_arr, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title("Imagem Equalizada")
    axes[1, 0].axis('off')

    # Histograma Eq
    axes[1, 1].bar(range(256), hist_eq, color='black', width=1.0)
    axes[1, 1].set_title("Histograma Equalizado")
    axes[1, 1].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

image=Image.open('arduino.jpeg')
image_gray = image_to_grayscale(image)
image_gray.save("2-arduino_gray_pil.jpeg")
gray_arr = np.array(image_gray)[:, :, 0]

hist_orig = get_histograma(gray_arr)
gray_eq_arr = equalizar(gray_arr)
hist_eq = get_histograma(gray_eq_arr)

plotar_resultados(gray_arr, hist_orig, gray_eq_arr, hist_eq)
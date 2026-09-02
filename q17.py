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

image_np=np.asarray(image_gray)
flat=image_np.flatten()
plt.hist(flat,bins=50)

def get_histogram(image_np,bins):
    histogram=np.zeros(bins)
    for pixel in image_np:
        histogram[pixel]+=1
    return histogram

hist=get_histogram(flat,256)

#plt.plot(hist)

def cum_sum(a):
    a=iter(a)
    b=[next(a)]

    for i in a:
        b.append(b[-1]+i)
    return np.array(b)

cs=cum_sum(hist)
#plt.plot(cs)

nj=(cs-cs.min())*255
N=cs.max()-cs.min()
cs=nj/N
#plt.plot(cs)

cs=cs.astype('uint8')

img_new=cs[flat]
#plt.hist(img_new,bins=50)

img_new=np.reshape(img_new,image_np.shape)

#plotar as imagens
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

fig.add_subplot(1,2,1)
plt.imshow(image, cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(img_new, cmap='gray')

plt.show(block=True)
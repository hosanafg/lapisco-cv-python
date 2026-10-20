import cv2
import numpy as np

image=cv2.imread("lagarto.jpeg")
image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

altura,largura=image_gray.shape
image_limiar=np.zeros((altura,largura),dtype=np.uint8)

limiar=100

for i in range(altura):
	for j in range(largura):
		px=image_gray[i,j]
		if px>limiar:
			image_limiar[i,j]=255
		else:
			image_limiar[i,j]=0

np.savetxt("matriz_limiarizada.txt",image_limiar)
print(f"Dimensões: {altura} x {largura}")

cv2.imshow("imagem limiarizada",image_limiar)
cv2.waitKey(0)
cv2.destroyAllWindows()


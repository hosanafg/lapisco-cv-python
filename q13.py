import cv2
import numpy as np
import matplotlib.pyplot as plt 

image_color = cv2.imread("lagarto.jpeg")
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
altura, largura = image_gray.shape

sobel_x_mask = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_mask = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

matriz_sobel_x = np.zeros((altura, largura), dtype=np.float32)
matriz_sobel_y = np.zeros((altura, largura), dtype=np.float32)
sobel_gradiente = np.zeros((altura, largura), dtype=np.float32)

for i in range (1,altura-1):
	for j in range(1,largura-1):
		regiao=image_gray[i-1:i+2,j-1:j+2].astype(np.float32)
		grad_x=np.sum(regiao*sobel_x_mask)
		grad_y=np.sum(regiao*sobel_y_mask)

		matriz_sobel_x[i,j]=grad_x
		matriz_sobel_y[i,j]=grad_y

		sobel_gradiente[i,j]=np.sqrt(grad_x**2 + grad_y**2)

sobel_x_norm = cv2.normalize(np.abs(matriz_sobel_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sobel_y_norm = cv2.normalize(np.abs(matriz_sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sobel_gradiente_norm = cv2.normalize(sobel_gradiente, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite('sobel_horizontal.jpg', sobel_x_norm)
cv2.imwrite('sobel_vertical.jpg', sobel_y_norm)
cv2.imwrite('sobel_final.jpg', sobel_gradiente_norm)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("1. Entrada (Tons de Cinza)")
plt.imshow(image_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("2. Sobel X (Bordas Verticais)")
plt.imshow(sobel_x_norm, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("3. Sobel Y (Bordas Horizontais)")
plt.imshow(sobel_y_norm, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("4. Sobel Magnitude (Resultado Final)")
plt.imshow(sobel_gradiente_norm, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

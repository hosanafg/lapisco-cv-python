#Abrir uma câmera, capturar uma imagem (frame), transforme em tom de cinza, visualizar imagema,
#de entrada, aplique o filtro de canny e visualize os resultados. Continue infinitamente capturando, 
#transformando em tom de cinza, aplicando canny e visualizando.

import cv2
import os 

pasta_destino="capturas"

if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

contador=1
cap=cv2.VideoCapture(0)

pressed_key=cv2.waitKey(1) & 0xFF

while(1):
    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_canny=cv2.Canny(frame_gray,30,100)
    
    #cv2.imshow('Frames_Gray',frame_gray)
    cv2.imshow('Frames_Canny',frame_canny)

    if pressed_key==ord('q'):
            break
    
    if pressed_key==ord('s'):
            nome_arquivo=f"Frame_{contador:02d}.jpeg"
            path_arquivo=os.path.join(pasta_destino, nome_arquivo)

            cv2.imwrite(path_arquivo, frame_canny)
            print(f"[DEBUG] Imagem salva: {path_arquivo}")
            contador+=1

cap.release()
cv2.destroyAllWindows()
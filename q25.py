"""
Crescimento de Regiões (Region Growing) com múltiplas sementes.
 
- Abre uma imagem colorida (640x480, objetos pretos em fundo branco)
- Converte para tons de cinza
- Cada clique do mouse define uma semente:
    1º clique -> região 1 (vermelho)
    2º clique -> região 2 (azul)
    3º clique -> região 3 (verde)
- Pixels que não pertencem a nenhum objeto permanecem com rótulo 0
 
 
Teclas: R = reiniciar | S = salvar resultado | ESC/Q = sair
"""
 
import sys
from collections import deque

from variables import *
from functions import *

import cv2
import numpy as np
  
# ----------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------
class SegmentadorInterativo:
    def __init__(self, img_bgr, max_regioes=3):
        self.cinza = converter_para_cinza(img_bgr)
        self.max_regioes = max_regioes
        self.reiniciar()
 
    def reiniciar(self):
        self.mapa_rotulos=np.zeros(self.cinza.shape, dtype=np.uint8)  #inicia o mapa_rotulos q vai ser passado na funcao crescimento_regioes
        self.proximo_rotulo = 1
        self.resultado = colorir_regioes(self.cinza, self.mapa_rotulos)
 
    def ao_clicar(self, evento, x, y):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if self.proximo_rotulo > self.max_regioes:
            print("Número máximo de regiões atingido (tecle R para reiniciar).")
            return
 
        n = crescimento_regioes(self.cinza, (x, y), self.proximo_rotulo,
                                self.mapa_rotulos)
        if n == 0:
            print("Esse pixel já pertence a uma região. Clique em outro objeto.")
            return
 
        print(f"Região {self.proximo_rotulo}: semente=({x}, {y}), {n} pixels")
        self.proximo_rotulo += 1
        self.resultado = colorir_regioes(self.cinza, self.mapa_rotulos)
 
    def executar(self, img_bgr):
        cv2.namedWindow("Region Growing")
        cv2.setMouseCallback("Region Growing", self.ao_clicar)
        while True:
            lado_a_lado = np.hstack([img_bgr, self.resultado])
            cv2.imshow("Region Growing", lado_a_lado)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("r"):
                self.reiniciar()
            if tecla == ord("s"):
                cv2.imwrite("resultado.png", self.resultado)
                np.save("mapa_rotulos.npy", self.mapa_rotulos)
                print("Salvo: resultado.png e mapa_rotulos.npy")
        cv2.destroyAllWindows()
 
 
def main():
    img = carregar_imagem(PATH)
    SegmentadorInterativo(img).executar(img)
 
if __name__ == "__main__":
    main()
 
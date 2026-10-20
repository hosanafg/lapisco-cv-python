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
 
import cv2
import numpy as np
 
LARGURA, ALTURA = 640, 480
LIMIAR = 30
CONECTIVIDADE = 8     # 4 ou 8 vizinhos

CORES = {
    1: (0, 0, 255),   # vermelho
    2: (255, 0, 0),   # azul
    3: (0, 255, 0),   # verde
}

# ----------------------------------------------------------------------------
# Imagem
# ----------------------------------------------------------------------------
def carregar_imagem(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir '{caminho}'.")
    if img.shape[:2] != (ALTURA, LARGURA):
        img = cv2.resize(img, (LARGURA, ALTURA), interpolation=cv2.INTER_NEAREST)
    return img
    
def converter_para_cinza(img_bgr):
    """Converte a imagem colorida para tons de cinza."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
 
# ----------------------------------------------------------------------------
# Region Growing
# ----------------------------------------------------------------------------
def obter_vizinhos(y, x, altura, largura, conectividade=8):
    """Retorna os vizinhos (y, x) válidos de um pixel."""
    if conectividade == 4:
        deslocamentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deslocamentos = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in deslocamentos:
        ny, nx = y + dy, x + dx                             #percorre os vizinhos da imagen
        if 0 <= ny < altura and 0 <= nx < largura:          #em caso do px estar na borda da img 
            yield ny, nx                                    #FIFO

 
def crescimento_regioes(cinza, semente, rotulo, mapa_rotulos,
                        limiar=LIMIAR, conectividade=CONECTIVIDADE):
    """
    Cresce uma região a partir de 'semente' = (x, y).
 
    Critério de similaridade: |I(p) - I(semente)| <= limiar
    Só considera pixels AINDA NAO ROTULADOS (mapa_rotulos inicial todo == 0).
    Atualiza 'mapa_rotulos' e retorna a quantidade de pixels da região.
    """
    altura, largura = cinza.shape
    x, y = semente
 
    if mapa_rotulos[y, x] != 0:
        return 0  # semente já pertence a outra região
 
    valor_semente = int(cinza[y, x])
    fila = deque([(y, x)])
    mapa_rotulos[y, x] = rotulo
    total = 1
 
    while fila:
        cy, cx = fila.popleft()
        for ny, nx in obter_vizinhos(cy, cx, altura, largura, conectividade):
            if mapa_rotulos[ny, nx] == 0 and abs(int(cinza[ny, nx]) - valor_semente) <= limiar:
                mapa_rotulos[ny, nx] = rotulo
                fila.append((ny, nx))
                total += 1
    return total
 
 
# ----------------------------------------------------------------------------
# Visualização
# ----------------------------------------------------------------------------
def colorir_regioes(cinza, mapa_rotulos, cores=CORES):
    """Gera imagem BGR: fundo em cinza, regiões pintadas com sua cor."""
    saida = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    for rotulo, cor in cores.items():
        saida[mapa_rotulos == rotulo] = cor
    return saida
 
 
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
    PATH = "/home/hosanafg/Documentos/pessoal/lap/img.png" #path no meu computador pessoal. mudar em caso de troca de dispostivos
    img = carregar_imagem(PATH)
    SegmentadorInterativo(img).executar(img)
 
if __name__ == "__main__":
    main()
 
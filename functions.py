import os
import cv2
import sys
import colorsys
from collections import deque
import numpy as np

from variables import *


# ----------------------------------------------------------------------------
# Carregar Imagem e Pré-Processamento
# ----------------------------------------------------------------------------
def carregar_imagem(caminho):
    image=cv2.imread(caminho)
    if image is None:
        raise FileNotFoundError(f"ERRO AO ABRIR '{caminho}'.")
    return image


def converter_para_cinza(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def binarizar_otsu(cinza):
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def detectar_bordas_canny(cinza, baixo=CANNY_BAIXO, alto=CANNY_ALTO):
    """
    Suaviza a imagem e aplica Canny. Em seguida faz um fechamento morfológico
    para unir pequenas falhas nas bordas, o que evita que um mesmo objeto
    seja contado como mais de um.
    """
    suave = cv2.GaussianBlur(cinza, (5, 5), 1.4)
    bordas = cv2.Canny(suave, baixo, alto)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel)

 
def encontrar_contornos(bordas, perimetro_minimo=PERIMETRO_MINIMO):
    """
    Aplica findContours na imagem de bordas.
    """
    resultado = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = resultado[-2]    # OpenCV 4 devolve 2 valores; OpenCV 3 devolve 3
    return [c for c in contornos if cv2.arcLength(c, True) >= perimetro_minimo]


def desenhar_contornos(img_bgr, contornos):
    saida = img_bgr.copy()
    cv2.drawContours(saida, contornos, -1, (0, 200, 0), 2)
    for i, c in enumerate(contornos, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(saida, str(i), (x + w // 2 - 8, y + h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(saida, f"Objetos: {len(contornos)}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    return saida
 

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
# Detecção automática de regiões
# ----------------------------------------------------------------------------
def encontrar_sementes_e_segmentar(binaria):  #subst dos cliques do mouse
    """
    Varre a imagem em ordem de linhas; todo pixel de objeto (valor 0) que
    ainda não tem rótulo é uma nova semente e dispara um region growing.
 
    Retorna (mapa_rotulos, total_objetos).
    """
    mapa_rotulos = np.zeros(binaria.shape, dtype=np.int32)  # 0 = fundo
    rotulo = 0
 
    for y, x in np.argwhere(binaria == 0):       # candidatos a semente. vai varrer a imagem e ver onde é 0 (argwhere)
        if mapa_rotulos[y, x] != 0:              # já pertence a uma região
            continue
        rotulo += 1
        # limiar=0: só entra quem tem exatamente o mesmo valor
        crescimento_regioes(binaria, (x, y), rotulo, mapa_rotulos, limiar=0)
    return mapa_rotulos, rotulo



# ----------------------------------------------------------------------------
# CRIAR E CÁLCULAR ÁREA: retângulos
# ----------------------------------------------------------------------------
def obter_retangulos(contornos):
    """
    Percorre todos os contornos e devolve o retângulo (x, y, largura, altura)
    de cada um, na mesma ordem da lista de contornos.
    """
    return [cv2.boundingRect(contorno) for contorno in contornos]
 
 
def desenhar_retangulos(img_bgr, retangulos):
    saida = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(retangulos, start=1):
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(saida, str(i), (x, max(y - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return saida
 

def calcular_areas(contornos):
    return [cv2.contourArea(contorno) for contorno in contornos]
 
 
def escrever_areas(img_bgr, retangulos, areas):
    saida = img_bgr.copy()
    altura = saida.shape[0]
    for (x, y, w, h), area in zip(retangulos, areas):
        ty = min(y + h + 20, altura - 6)
        cv2.putText(saida, f"A={area:.0f}", (x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    return saida
 
 
def imprimir_tabela(retangulos, areas):
    print(f"{'Obj':>3} | {'x':>4} {'y':>4} {'larg':>5} {'alt':>5} | {'área (px²)':>10}")
    print("=" *20 )
    for i, ((x, y, w, h), area) in enumerate(zip(retangulos, areas), start=1):
        print(f"{i:>3} | {x:>4} {y:>4} {w:>5} {h:>5} | {area:>10.1f}")


# ----------------------------------------------------------------------------
# Cores e visualização
# ----------------------------------------------------------------------------
def gerar_cores_distintas(n):
    """Gera n cores BGR bem espaçadas no matiz (HSV)."""
    cores=[]
    for i in range(n):
        r,g,b = colorsys.hsv_to_rgb(i / n, 0.9, 0.95)
        cores.append((int(b * 255), int(g * 255), int(r * 255)))
    return cores


def colorir_regioes(cinza, mapa_rotulos, cores=CORES):
    """Gera imagem BGR: fundo em cinza, regiões pintadas com sua cor."""
    saida = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    for rotulo, cor in cores.items():
        saida[mapa_rotulos == rotulo] = cor
    return saida

 
def colorir_regioes_automatico(mapa_rotulos, total, cor_fundo=(255, 255, 255)):
    saida = np.full((*mapa_rotulos.shape, 3), cor_fundo, dtype=np.uint8)
    for rotulo, cor in enumerate(gerar_cores_distintas(total), start=1):
        saida[mapa_rotulos == rotulo] = cor
    return saida

 
def escrever_numeros(img_bgr, mapa_rotulos, total):
    for rotulo in range(1, total + 1):
        ys, xs = np.nonzero(mapa_rotulos == rotulo)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(img_bgr, str(rotulo), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img_bgr


# ----------------------------------------------------------------------------
# Extração das subimagens
# ----------------------------------------------------------------------------
def calcular_caixa_delimitadora(mapa_rotulos, rotulo, margem=5):
    """
    Retorna (x0, y0, x1, y1) da menor caixa que contém a região 'rotulo',
    ampliada por 'margem' pixels e limitada às bordas da imagem.
    x1 e y1 são exclusivos (prontos para fatiar: img[y0:y1, x0:x1]).
    """
    altura, largura = mapa_rotulos.shape
    ys, xs = np.nonzero(mapa_rotulos == rotulo)
    x0 = max(xs.min() - margem, 0)
    y0 = max(ys.min() - margem, 0)
    x1 = min(xs.max() + 1 + margem, largura)
    y1 = min(ys.max() + 1 + margem, altura)
    return int(x0), int(y0), int(x1), int(y1)
 
 
def extrair_subimagem(img_bgr, mapa_rotulos, rotulo, margem=5):
    """
    Recorta o objeto 'rotulo' da imagem original.
    Tudo que está dentro da caixa mas NÃO pertence ao objeto vira branco.
    """
    x0, y0, x1, y1 = calcular_caixa_delimitadora(mapa_rotulos, rotulo, margem)
    recorte = img_bgr[y0:y1, x0:x1].copy()
    mascara = mapa_rotulos[y0:y1, x0:x1] == rotulo
    recorte[~mascara] = (255, 255, 255)
    return recorte
 
 
def extrair_todas_subimagens(img_bgr, mapa_rotulos, total, margem=5):
    """Retorna um dicionário {rótulo: subimagem} com todos os objetos."""
    return {r: extrair_subimagem(img_bgr, mapa_rotulos, r, margem)
            for r in range(1, total + 1)}
 
 
# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def salvar_subimagens(subimagens, pasta=PASTA_SAIDA):
    os.makedirs(pasta, exist_ok=True)
    for rotulo, sub in subimagens.items():
        cv2.imwrite(os.path.join(pasta, f"objeto_{rotulo}.png"), sub)
 
 
def mostrar_subimagens(subimagens, escala_minima=150):
    """
    Abre uma janela por subimagem.
    Subimagens muito pequenas são ampliadas só para exibição.
    """
    for i, (rotulo, sub) in enumerate(subimagens.items()):
        h, w = sub.shape[:2]
        if max(h, w) < escala_minima:
            f = escala_minima / max(h, w)
            sub = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        nome = f"Objeto {rotulo}"
        cv2.imshow(nome, sub)
        cv2.moveWindow(nome, 40 + 40 * (i % 10), 40 + 40 * (i % 10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calcular_bordas(cinza):
    """Mapa de bordas (Canny), apenas para visualização e validação."""
    suave = cv2.GaussianBlur(cinza, (5, 5), 1.5)
    return cv2.Canny(suave, CANNY_ALTO // 2, CANNY_ALTO)
 
 
def apoio_do_contorno(bordas, cx, cy, r, pontos=360, tolerancia=2):
    """
    Fração (0 a 1) dos pontos do contorno do círculo (cx, cy, r) que caem
    sobre pixels de borda (com uma pequena tolerância em pixels).
    Um círculo de verdade tem contorno quase todo sobre bordas; um falso
    positivo (cantos arredondados, arcos parciais) tem pouco apoio.
    """
    altura, largura = bordas.shape
    k = 2 * tolerancia + 1
    bordas_dil = cv2.dilate(bordas, np.ones((k, k), np.uint8))
 
    angulos = np.linspace(0, 2 * np.pi, pontos, endpoint=False)
    xs = np.round(cx + r * np.cos(angulos)).astype(int)
    ys = np.round(cy + r * np.sin(angulos)).astype(int)
 
    dentro = (xs >= 0) & (xs < largura) & (ys >= 0) & (ys < altura)
    if not dentro.any():
        return 0.0
    # Pontos fora da imagem contam como "sem apoio"
    apoiados = bordas_dil[ys[dentro], xs[dentro]] > 0
    return float(apoiados.sum()) / pontos


def detectar_circulo(cinza):
    """
    Aplica a Transformada de Hough para círculos e devolve (x, y, raio, apoio)
    do círculo mais forte e válido, ou None se não houver círculo.
    """
    suave = cv2.GaussianBlur(cinza, (9, 9), 2)
    circulos = cv2.HoughCircles(
        suave,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=cinza.shape[0],      # no máximo 1 círculo (distância mínima enorme)
        param1=CANNY_ALTO,
        param2=LIMIAR_ACUMULADOR,
        minRadius=RAIO_MIN,
        maxRadius=RAIO_MAX,
    )
    if circulos is None:
        return None
 
    bordas = calcular_bordas(cinza)
    melhor = None
    for cx, cy, r in circulos[0]:        # já vêm ordenados por votos
        apoio = apoio_do_contorno(bordas, cx, cy, r)
        if apoio >= APOIO_MINIMO and (melhor is None or apoio > melhor[3]):
            melhor = (int(round(cx)), int(round(cy)), int(round(r)), apoio)
    return melhor
 
 
# ----------------------------------------------------------------------------
# Visualização - CIRCULO
# ----------------------------------------------------------------------------
def desenhar_circulo(img_bgr, circulo):
    """Desenha contorno (verde) e centro (vermelho) do círculo detectado."""
    saida = img_bgr.copy()
    cx, cy, r, _ = circulo
    cv2.circle(saida, (cx, cy), r, (0, 255, 0), 3)
    cv2.circle(saida, (cx, cy), 3, (0, 0, 255), -1)
    return saida
 
 
def sinalizar_sem_circulo(img_bgr):
    """Escreve um aviso na imagem quando nenhum círculo é encontrado."""
    saida = img_bgr.copy()
    cv2.putText(saida, "Nenhum circulo detectado", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return saida
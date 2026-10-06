import os
import cv2
import numpy as np

# ====== Parâmetros ====== 
IMAGEM="output/lagarto.jpeg"
TOLERANCIA=40        
LARGURA=640
JANELA=7 
PASTA_SAIDA="/Users/hosana/Documents/lapisco-training-opencv-python/output"
# ========================

def carregar_imagem(caminho, largura=LARGURA):
    image=cv2.imread(caminho)
    if image is None:
        raise FileNotFoundError(f"ERRO AO ABRIR '{caminho}'.")

    escala=largura/image.shape[1]
    res=cv2.resize(image, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(res, (5, 5), 0)

def obter_clique(imagem, nome_janela="image"):
    """Mostra a imagem e devolve o clique do usuário como (linha, coluna)."""
    clique = []

    def callback(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            clique.append((y, x)) 
    cv2.namedWindow(nome_janela)
    cv2.setMouseCallback(nome_janela, callback)

    print("Clique sobre o objeto (ESC para sair).")
    while not clique:
        cv2.imshow(nome_janela, imagem)
        if cv2.waitKey(20) & 0xFF==27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelado pelo usuário.")

    cv2.destroyAllWindows()
    return clique[0]


def escolher_semente(B, clique, janela=JANELA):
    """Dentro de uma janela ao redor do clique, escolhe o pixel com MENOR canal B."""
    altura, largura=B.shape
    y,x=clique
    m=janela // 2
    y0, y1=max(0, y-m), min(altura, y+m+1)
    x0, x1 = max(0, x-m), min(largura, x+m+1)

    recorte = B[y0:y1, x0:x1]
    iy, ix = np.unravel_index(np.argmin(recorte), recorte.shape)
    return (y0 + iy, x0 + ix)


def calcular_candidatos(B, G, R, semente, tolerancia=TOLERANCIA):
    ref = np.array([B[semente], G[semente], R[semente]], dtype=np.float32)
    print(f"Referência (B, G, R): {ref}")

    img_f = np.stack([B, G, R], axis=-1).astype(np.float32)
    distancia = np.linalg.norm(img_f - ref, axis=-1)
    return distancia <= tolerancia


def crescer_regiao(candidatos, semente):
    regiao=np.zeros(candidatos.shape, dtype=bool)
    regiao[semente] = True
    kernel=np.ones((3, 3), np.uint8)
    iteracoes=0

    while True:
        vizinhos=cv2.dilate(regiao.astype(np.uint8), kernel).astype(bool) & ~regiao
        novos=vizinhos & candidatos
        if not novos.any():
            break
        regiao |= novos
        iteracoes += 1
    return regiao, iteracoes

def montar_sobreposicao(imagem, regiao, semente):
    overlay=imagem.copy()
    overlay[regiao] = (0, 255, 0)
    overlay = cv2.addWeighted(imagem, 0.5, overlay, 0.5, 0)
    cv2.circle(overlay, (semente[1], semente[0]), 4, (0, 0, 255), -1)
    return overlay

def mostrar(**imagens):
    for titulo, img in imagens.items():
        cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def salvar(pasta, prefixo, **imagens):
    """Salva cada imagem como <prefixo>_<nome>.png na pasta indicada."""
    os.makedirs(pasta, exist_ok=True)
    for nome, img in imagens.items():
        cv2.imwrite(os.path.join(pasta, f"{prefixo}_{nome}.png"), img)
    print("Imagens salvas em:", pasta)

def main():
    imagem=carregar_imagem(IMAGEM)
    B, G, R=cv2.split(imagem)
    clique=obter_clique(imagem)
    semente=escolher_semente(B, clique)
    print(f"Clique: {clique} | Semente: {semente}")

    candidatos=calcular_candidatos(B, G, R, semente)
    regiao, iteracoes=crescer_regiao(candidatos, semente)
    print(f"Região estabilizou após {iteracoes} iterações.")
    print(f"Pixels na região: {regiao.sum()}")

    resultado = (regiao * 255).astype(np.uint8)
    overlay = montar_sobreposicao(imagem, regiao, semente)

    mostrar(Canal_B=B, Canal_G=G, Canal_R=R,
            Regiao_segmentada=resultado, Sobreposicao=overlay)

    salvar(PASTA_SAIDA, "Q24_Lagarto",
           original=imagem, canal_B=B, canal_G=G, canal_R=R,
           regiao_segmentada=resultado, sobreposicao=overlay)

if __name__ == "__main__":
    main()
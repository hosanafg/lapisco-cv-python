#PATH = "/home/hosanafg/Documentos/pessoal/lap/img.png" #path no meu computador pessoal. mudar em caso de troca de dispostivos
PATH="/Users/hosana/Documents/lapisco-training-opencv-python/img.png"
PATH_COLOR_IMG="/Users/hosana/Documents/lapisco-training-opencv-python/output/arduino.jpeg"
PASTA_SAIDA = "/Users/hosana/Documents/lapisco-training-opencv-python/output/subimagens"

LARGURA, ALTURA = 640, 480
LIMIAR = 30
CONECTIVIDADE = 8     # 4 ou 8 vizinhos

CORES = {
    1: (0, 0, 255),   # vermelho
    2: (255, 0, 0),   # azul
    3: (0, 255, 0),   # verde
}

# Parâmetros do Canny / Hough
CANNY_ALTO = 100         
CANNY_BAIXO = 50
LIMIAR_ACUMULADOR = 40   # quanto menor, mais círculos (e falsos positivos)
RAIO_MIN = 10
RAIO_MAX = 0             # 0 = sem limite
APOIO_MINIMO = 0.75     
PERIMETRO_MINIMO = 20

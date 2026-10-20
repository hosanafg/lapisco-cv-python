from functions import *
from variables import *

import gc
import os
import sys
import tracemalloc

PASTA_SAIDA = "output/subimagens_contornos"
LADO_MINIMO = 20     
MAX_NA_GRADE = 30 

# ----------------------------------------------------------------------------
# Questão 33: subimagens
# ----------------------------------------------------------------------------
def filtrar_retangulos(retangulos, lado_minimo=LADO_MINIMO):
    """Descarta retângulos muito pequenos (ruído). Com 0, mantém todos."""
    return [(x, y, w, h) for (x, y, w, h) in retangulos
            if w >= lado_minimo and h >= lado_minimo]
 
 
def extrair_subimagem(img, retangulo):
    """
    Recorta a região do retângulo (x, y, largura, altura).
 
    .copy() sem ele, o recorte é só uma "janela" que
    aponta para a imagem original e a mantém viva na memória enquanto o
    recorte existir.
    """
    x, y, w, h = retangulo
    return img[y:y + h, x:x + w].copy()
 
 
def salvar_subimagem(sub, pasta, indice):
    cv2.imwrite(os.path.join(pasta, f"objeto_{indice}.png"), sub)
 
 
def montar_grade(subimagens, lado=110, colunas=6):
    """Junta miniaturas lado a lado em uma única imagem (uma só janela)."""
    miniaturas = []
    for sub in subimagens:
        h, w = sub.shape[:2]
        f = lado / max(h, w)
        mini = cv2.resize(sub, (max(int(w * f), 1), max(int(h * f), 1)))
        celula = np.full((lado, lado, 3), 255, dtype=np.uint8)
        celula[:mini.shape[0], :mini.shape[1]] = mini
        cv2.rectangle(celula, (0, 0), (lado - 1, lado - 1), (180, 180, 180), 1)
        miniaturas.append(celula)
 
    while len(miniaturas) % colunas:                 # completa a última linha
        miniaturas.append(np.full((lado, lado, 3), 255, dtype=np.uint8))
    linhas = [np.hstack(miniaturas[i:i + colunas])
              for i in range(0, len(miniaturas), colunas)]
    return np.vstack(linhas)
 
 
# ----------------------------------------------------------------------------
# Questão 34: medição de memória
# ----------------------------------------------------------------------------
def memoria_atual_mb():
    """Memória Python/NumPy atualmente alocada em MB."""
    atual, _ = tracemalloc.get_traced_memory()
    return atual / 1024 / 1024
 
 
def processar_subimagens(img, retangulos, pasta, acumular):
    """
    Gera todas as subimagens e mede a memória após cada iteração.
 
    acumular=True : guarda cada subimagem em uma lista (a memória só cresce)
    acumular=False: salva em disco e descarta com del (a memória fica estável)
 
    Retorna (lista_de_memoria_por_iteracao, subimagens_guardadas).
    """
    os.makedirs(pasta, exist_ok=True)
    base = memoria_atual_mb()
    medidas = []
    guardadas = []
 
    for i, retangulo in enumerate(retangulos, start=1):
        sub = extrair_subimagem(img, retangulo)
        salvar_subimagem(sub, pasta, i)
 
        if acumular:
            guardadas.append(sub)         # a lista mantém o objeto vivo
        else:
            del sub                       # nenhuma referência sobra: memória liberada
        medidas.append(memoria_atual_mb() - base)
 
    return medidas, guardadas
 
 
def resumir(nome, medidas):
    """Imprime memória no início, meio e fim das iterações."""
    n = len(medidas)
    print(f"{nome:<10} | 1ª iter: {medidas[0]:7.3f} MB | "
          f"meio: {medidas[n // 2]:7.3f} MB | última: {medidas[-1]:7.3f} MB")
 
 
# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    img = carregar_imagem(PATH_COLOR_IMG)
    cinza = converter_para_cinza(img)
    bordas = detectar_bordas_canny(cinza)
    contornos = encontrar_contornos(bordas)
    retangulos = filtrar_retangulos(obter_retangulos(contornos))
    print(f"Contornos encontrados: {len(contornos)} | retângulos usados: {len(retangulos)}")
 
    # liberar Bordas e Contornos
    del cinza, bordas, contornos
 
    tracemalloc.start()
 
    medidas_acum, subs = processar_subimagens(img, retangulos, PASTA_SAIDA, acumular=True)
    resumir("ACUMULANDO", medidas_acum)
    grade = montar_grade(subs[:MAX_NA_GRADE])   # usa as guardadas para a visualização
    del subs                                    # solta a lista inteira de uma vez
    gc.collect()
    print(f"Após del + gc.collect(): {memoria_atual_mb():.3f} MB acima do início "
          "(restam só a grade e as medidas)")
 
    medidas_lib, _ = processar_subimagens(img, retangulos, PASTA_SAIDA, acumular=False)
    resumir("LIBERANDO", medidas_lib)
 
    tracemalloc.stop()
 
    cv2.imshow(f"Primeiras {min(MAX_NA_GRADE, len(retangulos))} subimagens", grade)
    cv2.waitKey(0)
    cv2.destroyAllWindows()     
 
 
if __name__ == "__main__":
    main()
 
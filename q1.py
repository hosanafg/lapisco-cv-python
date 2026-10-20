from functions import *
from variables import * 

def main():
    image=carregar_imagem(PATH)
    cv2.imwrite("arduino_copy.jpeg",image)
    cv2.imshow("imagecopy",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="main":
    main()


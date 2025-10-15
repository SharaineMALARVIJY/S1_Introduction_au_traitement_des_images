# Sharaine MALARVIJY 21206543
#%% Fonctions 

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters


def affiche(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im)

def affiche_gray(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im, cmap="gray")

def clip_int8(rgb):
    return np.uint8(np.clip(rgb, 0, 255))


#%% Test ici
if __name__ == "__main__" :
    # Exercice 1
    
    im_refocus_1 = np.array(Image.open("../Images_TP/Refocus_1.png"))
    im_refocus_2 = np.array(Image.open("../Images_TP/Refocus_2.png"))
    im_refocus_3 = np.array(Image.open("../Images_TP/Refocus_3.png"))

    affiche(im_refocus_1)
    affiche(im_refocus_2)
    affiche(im_refocus_3)

    claire3 = im_refocus_3[:125, :]
    affiche(claire3)

    claire2 = im_refocus_2[125:200, :]
    affiche(claire2)

    claire1 = im_refocus_1[200:,:]
    affiche(claire1)

    new_a = np.zeros_like(im_refocus_1)
    new_a[200:,:] = claire1
    new_a[125:200, :] = claire2
    new_a[:125, :] = claire3

    affiche(new_a)

    #%% Exercice 2


    im_mosaic_1 = np.array(Image.open("../Images_TP/Mosaic_1.png"))
    im_mosaic_2 = np.array(Image.open("../Images_TP/Mosaic_2.png"))

    affiche(im_mosaic_1)
    affiche(im_mosaic_2)

    #%% Exercice 3

    im_resize = np.array(Image.open("../Images_TP/Resize.png"))
    im_resize_gray = np.array(Image.open("../Images_TP/Resize.png").convert("L"))

    affiche(im_resize)
    grad = filters.sobel(im_resize_gray)
    plt.figure()
    plt.title("Heat map")
    plt.imshow(grad, cmap="hot")
    

    def resizing(a):
        x_a, y_a = a.shape[1], a.shape[0]
        new_a = np.zeros_like(a)
        path = []

        grad = filters.sobel(a)

        x_start = np.argmin(grad[0, :]) 
        next_point = [0, x_start]   #premier point

        for i in range(x_a):
                y, x = next_point[i, 0], next_point[i, 1]

                voisin_bas = ([grad[y+1, x-1], grad[y, x+1], grad[y+1, x+1]])
                voisin_min = np.argmin(voisin_bas)
                if voisin_min==0 :
                    path.append([y+1, x-1])
                elif voisin_min==1:
                    path.append([y, x+1])
                else : #voisin_min==2
                    path.append([y+1, x+1])

                next_point.append(path[-1])
        return path
    
    im_path = np.zeros_like(im_resize)
    path = resizing(im_resize_gray)

    


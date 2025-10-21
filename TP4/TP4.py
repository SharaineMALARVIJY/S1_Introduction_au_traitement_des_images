# Sharaine MALARVIJY 21206543
#%% Fonctions 
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage.color import rgb2gray


def affiche(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im)

def affiche_gray(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im, cmap="gray")

def affiche_3(im0, im1, im2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(im0)

    plt.subplot(1, 3, 2)
    plt.imshow(im1)

    plt.subplot(1, 3, 3)
    plt.imshow(im2)
    plt.show()
     

#%% Test ici
if __name__ == "__main__" :
    # Exercice 1
    
    im_monalisa_square = np.array(Image.open("../Images_TP/MonaLisa_square.jpg").convert("RGB"))
    affiche(im_monalisa_square)
    def photomation(a):
        x_a, y_a = a.shape[0], a.shape[1]
        new_a = np.zeros_like(a)
        H_half = y_a//2
        L_half = x_a//2
        for y in range(0, y_a, 2):
            for x in range(0, x_a, 2):
                x_half = x//2
                y_half = y//2

                new_a[y_half,        x_half] = a[y, x]
                new_a[y_half,        L_half+x_half] = a[y, x+1]
                new_a[H_half+y_half, x_half] = a[y+1, x]
                new_a[H_half+y_half, L_half+x_half] = a[y+1, x+1]

        return new_a
    
    im = photomation(im_monalisa_square)
    for i in range(12):
        affiche(im)
        im = photomation(im)
    
    #%% Exercice 2

    im_lena = np.array(Image.open("../Images_TP/Lena.jpg").convert("RGB"))

    affiche(im_lena)

    # Methode a réaliser
    # prendre le premier RGB et parcourir l'image 
    # modifier tous les triplets qui sont pareil au RGB inital 
    


    def all_rgb(a):
        x_a, y_a = a.shape[0], a.shape[1]
        new_a = np.zeros_like(a)
        used_pixel = []

        used_pixel.append(a[0, 0])
        new_a[0, 0] = a[0, 0]

        for y in range(1, y_a//3):
            for x in range(1, x_a//3):

                arr = np.array(used_pixel, dtype=np.uint8)
                exists = np.any(np.all(arr == a[y, x], axis=1))

                if not exists:
                    used_pixel.append(a[y, x])
                    new_a[y, x] = a[y, x]
                else:
                    new_a[y, x] = np.array([0, 255, 0], dtype=np.uint8)

        return new_a
    
    affiche(all_rgb(im_lena))

    #%% Exercice 4

    # segmentation
    # region proc simal ?


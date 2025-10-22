# Sharaine MALARVIJY 21206543
#%% Fonctions 
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage.color import rgb2gray
import time 

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

def clip_int8(rgb):
    return np.uint8(np.clip(rgb, 0, 255))

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

#%% Test ici
if __name__ == "__main__" :
    # Exercice 1
    
    im_monalisa_square = np.array(Image.open("../Images_TP/MonaLisa_square.jpg").convert("RGB"))
    affiche(im_monalisa_square)
    
    im = photomation(im_monalisa_square)
    im_list = []
    for i in range(12):
        im_list.append(im)
        im = photomation(im)

    for i in range(0, 12, 3):
        affiche_3(im_list[i], im_list[i+1], im_list[i+2])



    
    #%% Exercice 2

    im_lena = np.array(Image.open("../Images_TP/Lena.jpg").convert("RGB"))

    affiche(im_lena, "Image original")

    def all_RGB(a):
        x_a, y_a = a.shape[1], a.shape[0]
        new_a = np.zeros_like(a)
        used_pixel = set() 
        alea = 10
        for y in range(y_a):
            for x in range(x_a):
                pixel = tuple(a[y, x])
                while pixel in used_pixel:               
                    outR = np.clip(int(pixel[0])+np.random.randint(-alea, alea), 0, 255)
                    outG = np.clip(int(pixel[1])+np.random.randint(-alea, alea), 0, 255)
                    outB = np.clip(int(pixel[2])+np.random.randint(-alea, alea), 0, 255)
                    pixel = (outR, outG, outB)
                new_a[y, x] = np.array([pixel[0], pixel[1], pixel[2]], dtype=np.uint8)
                used_pixel.add(pixel)
        return new_a
    
    def rgb_redondant_vert(a):
        """Si un pixel est redondant il sera remplacer par un pixel vert"""
        x_a, y_a = a.shape[1], a.shape[0]
        new_a = np.zeros_like(a)
        used_pixel = set()  
        doublon = False

        for y in range(y_a):
            for x in range(x_a):
                pixel = tuple(a[y, x]) 

                if pixel not in used_pixel:
                    new_a[y, x] = a[y, x]   
                    used_pixel.add(pixel)  
                else:
                    doublon = True
                    new_a[y, x] = np.array([0, 255, 0], dtype=np.uint8)

        return new_a, doublon


    im_lena_all_rgb = all_RGB(im_lena)
    affiche(im_lena_all_rgb, "Après all rgb")

    im_lena_vert, doublon = rgb_redondant_vert(im_lena_all_rgb)

    if doublon:
        affiche(im_lena_vert, "Echec")
    else:
        affiche(im_lena[330:400, 350:420], "Image original")
        affiche(im_lena_all_rgb[330:400, 350:420], "Après all rgb")

    t1 = time.time()
    t2 = time.time()
    print(f"{t2-t1:.2f}s de calcul")

    #%% Exercice 4

    # segmentation
    # region proc simal ?

    

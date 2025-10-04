# Sharaine MALARVIJY 21206543
#%% Fonctions 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import random as rd

def affiche(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im)

def affiche_gray(im, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(im, cmap="gray")

def convolution(im, filtre=[1,1,1,1,1,1,1,1,1]):
    a = np.asarray(im)
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    #On copie les bords de l'image original
    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]
    
    for y in range(1, y_a-1):
        for x in range(1, x_a-1):
                matrice = ([a[y-1, x-1], a[y-1, x], a[y-1, x+1],
                            a[y, x-1], a[y, x], a[y, x+1],
                            a[y+1, x-1], a[y, x+1], a[y+1, x+1]])
                for i in range(len(matrice)):    
                    val += matrice[i]*filtre[i]
                new_a[y, x] = val
    return new_a

def bruit_sel_1x1(a, taux_sel=0.15):
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    for y in range(0, y_a):
        for x in range(0, x_a):
                r = rd.random()
                if r > taux_sel :
                    new_a[y, x] = a[y, x]
                else:
                    new_a[y, x] = np.array([255,255,255])
    return new_a

def bruit_sel_2x2(a, taux_sel=0.05):
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    for y in range(0, y_a, 2):
        for x in range(0, x_a, 2):
                r = rd.random()
                if r > taux_sel :
                    new_a[y-1:y+1, x-1:x+1] = a[y-1:y+1, x-1:x+1]
                else:
                    new_a[y-1:y+1, x-1:x+1] = np.array([255,255,255])
    return new_a

def filtre_median(a):
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    #On copie les bords de l'image original
    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]

    for y in range(1, y_a-1):
        for x in range(1, x_a-1):
            for c in range(3):
                matrice = np.array([a[y-1, x-1][c], a[y-1, x][c], a[y-1, x+1][c],
                            a[y, x-1][c], a[y, x][c], a[y, x+1][c],
                            a[y+1, x-1][c], a[y, x+1][c], a[y+1, x+1][c]])
                new_a[y, x][c] = np.sort(matrice)[4]      
    return new_a

def mask_clown(i):
    i[80:95, 15:30]  = 0
    i[30:45, 100:115] = 0
    i[45:55, 40:50] = 0
    i[70:80, 80:90] = 0  
    return i

def mask_im_noise(i):
    i[120:130, :65]  = 0
    i[120:130, 185:] = 0
    return i

def fft(im, mask=""):
    i = np.fft.fft2(im)
    i = np.abs(i)
    i = np.log(i)
    i = np.fft.fftshift(i)
    if mask=="noise":
        mask_im_noise(i)
    if mask=="clown":
        mask_clown(i)
    return i


def fft_filter(im, mask=""):
    i_noisy = np.fft.fft2(im)
    i_noisy = np.fft.fftshift(i_noisy)

    if mask=="noise":
        mask_im_noise(i_noisy)
    if mask=="clown":
        mask_clown(i_noisy)

    i_noisy = np.fft.ifftshift(i_noisy)
    i_noisy = np.fft.ifft2(i_noisy)
    return np.abs(i_noisy)

im_lena = np.array(Image.open("../Images_TP/Lena.jpg"))
im_noise = np.array(Image.open("../Images_TP/noise.tif"))
im_clown = np.array(Image.open("../Images_TP/clown.tif"))

im_noisy_lena = np.array(Image.open("../Images_TP/noisy_Lena.png"))


#Test ici
if __name__ == "__main__" :
    #%% Exercice 1
    
    im_noisy_lena_maison = bruit_sel_1x1(im_lena)
    im_median = []
    im_median.append(filtre_median(im_noisy_lena_maison))
    im_median.append(filtre_median(im_median[0]))
    im_median.append(filtre_median(im_median[1]))

    affiche(im_lena, "Image Original de Lena")
    affiche(im_noisy_lena_maison, "Filtre sel fait maison")

    plt.figure()
    for i in range(0, 3):
        affiche(im_median[i], title=f"Filtre médian {i+1}")


    #%% Exercice 2

    affiche_gray(im_noise)
    affiche_gray(fft(im_noise, "noise"))
    affiche_gray(fft_filter(im_noise, "noise"))

    affiche_gray(im_clown)
    affiche_gray(fft(im_clown, "clown"))
    affiche_gray(fft_filter(im_clown, "clown"))

    #%% Exercice 3

    affiche(im_noisy_lena)
    affiche(im_noisy_lena_maison)



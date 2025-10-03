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

def bruit_sel(a, taux_sel=0.05):
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

def fft(im):
    i = np.fft.fft2(im)
    i = np.abs(i)
    i = np.log(i)
    i = np.fft.fftshift(i)
    # i[120:130, :65]  = 255
    # i[120:130, 185:] = 255
    return i


def filtering_im_noise(im):
    i_noisy = np.fft.fft2(im)
    i_noisy = np.fft.fftshift(i_noisy)
    i_noisy[120:130, :65]  = 255
    i_noisy[120:130, 185:] = 255
    i_noisy = np.fft.ifftshift(i_noisy)
    i_noisy = np.fft.ifft2(i_noisy)
    return np.abs(i_noisy)

def filtering_im_noise(im):
    i_noisy = np.fft.fft2(im)
    i_noisy = np.fft.fftshift(i_noisy)
    i_noisy[120:130, :65]  = 255
    i_noisy[120:130, 185:] = 255
    i_noisy = np.fft.ifftshift(i_noisy)
    i_noisy = np.fft.ifft2(i_noisy)
    return np.abs(i_noisy)

im_lena = np.array(Image.open("../Images_TP/Lena.jpg"))
im_noise = np.array(Image.open("../Images_TP/noise.tif"))
im_clown = np.array(Image.open("../Images_TP/clown.tif"))

im_noisy_lena = np.array(Image.open("../Images_TP/noisy_Lena.png"))

#%% Exercice 1
im_noisy_lena_maison = bruit_sel(im_lena)
im_median_1 = filtre_median(im_noisy_lena_maison)
im_median_2 = filtre_median(im_median_1)
im_median_3 = filtre_median(im_median_2)
#%%
affiche(im_lena, "Image Original de Lena")
affiche(im_noisy_lena_maison, "Filtre sel fait maison")

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im_median_1)
plt.title("Filtre médian 1")
plt.subplot(1, 3, 2)
plt.imshow(im_median_2)
plt.title("Filtre médian 2")
plt.subplot(1, 3, 3)
plt.imshow(im_median_3)
plt.title("Filtre médian 3")

#%% Exercice 2

affiche_gray(im_noise)
affiche_gray(fft(im_noise))
affiche_gray(filtering_im_noise(im_noise))

affiche_gray(im_clown)
affiche_gray(fft(im_clown))

#%% Exercice 3







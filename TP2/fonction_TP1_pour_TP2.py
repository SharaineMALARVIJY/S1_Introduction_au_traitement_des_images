from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, threshold_local

"""
Pour tout le code x est sur la vertical et y est sur l'horizontal
a <=> array

J'ai eu des problème sur les calculs avec le uint8 d'où l'existance des
fonctions addi_uint et sous_uint 

"""

def convertion_rouge(im):
    a = np.asarray(im)
    r_a = np.zeros_like(a)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            r_a[i][j] = np.array((a[i][j][0], 0, 0, a[i][j][3]))

    plt.figure()
    plt.title("convertion_rouge")
    plt.imshow(Image.fromarray(r_a))
    return Image.fromarray(r_a)


def convertion_negatif(im):
    a = np.asarray(im)
    n_a = np.zeros_like(a)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            n_a[i][j] = np.array((255-a[i][j][0], 255-a[i][j][1], 255-a[i][j][2], a[i][j][3]))

    plt.figure()
    plt.title("convertion_negatif")
    plt.imshow(Image.fromarray(n_a))
    return Image.fromarray(n_a)



def max_255(rgb):
    if rgb > 255 :
        return 255
    return rgb



def sepia(im):
    a = np.asarray(im)
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            outR = max_255(.393*a[i][j][0] + .769*a[i][j][1] + .189*a[i][j][2])
            outG = max_255(.349*a[i][j][0] + .686*a[i][j][1] + .168*a[i][j][2])
            outB = max_255(.272*a[i][j][0] + .534*a[i][j][1] + .131*a[i][j][2])
            new_a[i][j] = np.array((outR, outG, outB, a[i][j][3]))

    plt.figure()
    plt.title("sepia")
    plt.imshow(Image.fromarray(new_a))
    return Image.fromarray(new_a)

def calcul_contrast(rgb):
    if rgb < 30 :
        return 0
    if rgb > 225 :
        return 255
    return (255/195)*(rgb-30)+.5

def contrast(im):
    a = np.asarray(im)
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            outR = calcul_contrast(a[i][j][0])
            outG = calcul_contrast(a[i][j][1])
            outB = calcul_contrast(a[i][j][2])
            new_a[i][j] = np.array((outR, outG, outB, a[i][j][3]))
    
    plt.figure()
    plt.title("contrast")
    plt.imshow(Image.fromarray(new_a))

    

def seuillage_couleur(im):
    a = np.asarray(im)
    grey_img = greyscale(im, graph=False)
    grey_img = np.asarray(grey_img)
    
    seuil_otsu = threshold_otsu(grey_img)
    im_otsu = (a >= seuil_otsu) * 255
    
    plt.figure()
    plt.title("Seuillage_couleur - Otsu")
    plt.imshow(im_otsu)
    plt.show()

def seuillage_couleur_individuel(im):
    a = np.asarray(im)
    im_otsu = np.zeros_like(a)

    for i in range(3): 
        seuil = threshold_otsu(a[..., i])
        im_otsu[..., i] = (a[..., i] >= seuil) * 255
    
    plt.figure()
    plt.title("Seuillage_couleur_individuel - Otsu")
    plt.imshow(im_otsu)
    plt.show()


def flip(im):
    a = np.asarray(im)
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            new_a[x_a-1-i][y_a-1-j] = np.array((a[i][j][0], a[i][j][1], a[i][j][2], a[i][j][3]))
    
    plt.figure()
    plt.title("flip")
    plt.imshow(Image.fromarray(new_a))
    return Image.fromarray(new_a)


def border(im, bord = 5):
    a = np.asarray(im)
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            if (i<bord or i>=x_a-bord or j<bord or j>=y_a-bord):
                new_a[i][j] = np.array((0, 0, 255, a[i][j][3]))
            else: 
                new_a[i][j] = np.array((a[i][j][0], a[i][j][1], a[i][j][2], a[i][j][3]))
    
    plt.figure()
    plt.title("border")
    plt.imshow(Image.fromarray(new_a))
    return Image.fromarray(new_a)

def rgb(rgb):
    return np.uint8(np.clip(rgb, 0, 255))
    
def addi_uint(a, b):
    return np.uint8(rgb(int(a)+int(b)))

def sous_uint(a, b):
    return np.uint8(rgb(int(a)-int(b)))

def relief(im, bord = 10):
    a = np.asarray(im)
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            outR = a[i][j][0]
            outG = a[i][j][1]
            outB = a[i][j][2]
            outA = a[i][j][3]
            if (i<bord and i<=j and i<=y_a-j):
                haut = 65
                new_a[i][j] = np.array((addi_uint(outR, haut), addi_uint(outG, haut), addi_uint(outB, haut), outA))
            elif (i>=x_a-bord and y_a-j>=x_a-i and j>=x_a-i):
                bas = 65
                new_a[i][j] = np.array((sous_uint(outR, bas), sous_uint(outG, bas), sous_uint(outB, bas), outA))
            elif (j<bord or j>=y_a-bord):
                c = 40
                new_a[i][j] = np.array((sous_uint(outR, c), sous_uint(outG, c), sous_uint(outB, c), outA))
            else: 
                new_a[i][j] = np.array((outR, outG, outB, outA))

    plt.figure()
    plt.title("relief")
    plt.imshow(Image.fromarray(new_a))
    return Image.fromarray(new_a)









def accentuation(im):
    a = np.asarray(im)
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    #On copie les bords de l'image original
    new_a[0, :, :]  = a[0, :, :]
    new_a[-1, :, :] = a[-1, :, :]
    new_a[:, 0, :]  = a[:, 0, :]
    new_a[:, -1, :] = a[:, -1, :]

    for x in range(1, x_a-1):
        for y in range(1, y_a-1):
            for c in range(3):  
                val = ( 3*int(a[x, y, c])
                    +   (- int(a[x-1, y, c]) - int(a[x+1, y, c])
                         - int(a[x, y-1, c]) - int(a[x, y+1, c]))/2 )
                new_a[x, y, c] = rgb(val)


    plt.figure()
    plt.title("accentuation")
    plt.imshow(Image.fromarray(new_a))

def gradient(im):
    a = greyscale(im, graph=False)
    a = np.asarray(a)
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    #On copie les bords de l'image original
    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]

    for x in range(1, x_a-1):
        for y in range(1, y_a-1):
                val = ( - int(a[x-1, y-1]) + int(a[x-1, y+1])
                        - 2*int(a[x, y-1]) + 2*int(a[x, y+1])
                        - int(a[x+1, y-1]) + int(a[x+1, y+1]) )
                new_a[x, y] = rgb(val)

    plt.figure()
    plt.title("gradient")
    plt.imshow(Image.fromarray(new_a), 'grey')

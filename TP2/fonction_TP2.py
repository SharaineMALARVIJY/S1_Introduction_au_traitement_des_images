# Sharaine MALARVIJY 21206543
#%% Fonctions 

from PIL import Image, ImageOps
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
    """Bonne base pour les autres code mais pas utiliser au final"""
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
    """J'ai essayer de reproduire le bruit de noisy_Lena.png mais ça servait à rien"""
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
                            a[y+1, x-1][c], a[y+1, x+1][c], a[y+1, x+1][c]])
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

def fusion(im, drapeau):
    drapeau = drapeau.resize(im.size)

    np_eiffel = np.array(im).astype(int)
    np_flag = np.array(drapeau).astype(int)

    new_a = (0.5 * np_flag + 0.5 * np_eiffel).astype(np.uint8)
    new_im = Image.fromarray(new_a)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title("Image Original")

    plt.subplot(1, 3, 2)
    plt.imshow(drapeau)
    plt.title("Drapeau")

    plt.subplot(1, 3, 3)
    plt.imshow(new_im)
    plt.title("Fusion")
    plt.show()

## Fonction modifier du TP1 ##############################################################
# Quelques fonction ne sont pas utiliser car pas assez performant sur le débruitage des images
def clip_int8(rgb):
    return np.uint8(np.clip(rgb, 0, 255))

def normalisation(I, max=255):
    x_a, y_a = I.shape[0], I.shape[1]
    I_norm = np.zeros_like(I)

    LUT = np.zeros(256)
    I_min = np.min(I)
    I_max = min(max, np.max(I))
    for i in range(256):
        LUT[i] = clip_int8(255*( int(i)- int(I_min))/( I_max - I_min ))

    for x in range(x_a):
        for y in range(y_a):
            I_norm[x, y] = clip_int8(LUT[I[x,y]])
    return I_norm


def greyscale(a, graph=True):
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros((x_a, y_a), dtype = 'uint8')

    for i in range(x_a):
        for j in range(y_a):
            grey = (.299*a[i][j][0] + .587*a[i][j][1] + .114*a[i][j][2])
            new_a[i][j] = np.array(grey)

    if graph:
        plt.figure()
        plt.title("greyscale")
        plt.imshow(Image.fromarray(new_a), 'grey')
    return new_a

def pixeliser(a, ordinal=2):
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    for x in range(0, x_a, ordinal):
        for y in range(0, y_a, ordinal):
            x_end = min(x + ordinal, x_a)
            y_end = min(y + ordinal, y_a)

            mR = np.mean(a[x:x_end, y:y_end, 0])
            mG = np.mean(a[x:x_end, y:y_end, 1])
            mB = np.mean(a[x:x_end, y:y_end, 2])

            new_a[x:x_end, y:y_end] = np.array([mR, mG, mB])
    return new_a

def lissage(a):
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    for x in range(1, x_a-1):
        for y in range(1, y_a-1):
            
            mR = np.mean(a[x-1:x+2, y-1:y+2, 0])
            mG = np.mean(a[x-1:x+2, y-1:y+2, 1])
            mB = np.mean(a[x-1:x+2, y-1:y+2, 2])

            new_a[x, y] = np.array([mR, mG, mB])

    #On copie les bords de l'image original
    new_a[0, :, :]  = a[0, :, :]
    new_a[-1, :, :] = a[-1, :, :]
    new_a[:, 0, :]  = a[:, 0, :]
    new_a[:, -1, :] = a[:, -1, :]

    return new_a

def lissage_gros(a, fenetre= 3):
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    for x in range(1, x_a-1):
        for y in range(1, y_a-1):
            mR = np.mean(a[x-fenetre:x+1+fenetre, y-fenetre:y+fenetre+1, 0])
            mG = np.mean(a[x-fenetre:x+1+fenetre, y-fenetre:y+1+fenetre, 1])
            mB = np.mean(a[x-fenetre:x+1+fenetre, y-fenetre:y+1+fenetre, 2])

            new_a[x, y] = np.array([mR, mG, mB])

    #On copie les bords de l'image original
    new_a[0, :, :]  = a[0, :, :]
    new_a[-1, :, :] = a[-1, :, :]
    new_a[:, 0, :]  = a[:, 0, :]
    new_a[:, -1, :] = a[:, -1, :]

    return new_a

def accentuation(a):
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
                new_a[x, y, c] = clip_int8(val)

    return new_a

def gradient(a):
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
                new_a[x, y] = clip_int8(val)

    return new_a

from skimage.filters import threshold_otsu, threshold_local

def seuillage(im):
    grey_img = greyscale(im, graph=False)
    grey_img = np.asarray(grey_img)

    seuil_otsu = threshold_otsu(grey_img)
    im_otsu = (grey_img >= seuil_otsu) * 255

    block_size = 41 #131 # taille du voisinage 
    local_thresh = threshold_local(grey_img, block_size, offset=10) 
    im_adapt = (grey_img >= local_thresh) * 255
    return im_adapt

def calcul_contrast(rgb):
    if rgb < 30 :
        return 0
    if rgb > 225 :
        return 255
    return (255/195)*(rgb-30)+.5

def contrast(a):
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)

    for i in range(x_a):
        for j in range(y_a):
            outR = calcul_contrast(a[i][j][0])
            outG = calcul_contrast(a[i][j][1])
            outB = calcul_contrast(a[i][j][2])
            new_a[i][j] = np.array((outR, outG, outB))
    
    return new_a

def histogramme(I_hist, title="Histogramme"):
    plt.figure()
    plt.title(title)
    plt.hist(I_hist.ravel(), bins=256, color='black')

##################################################################

def lissage_adaptatif(a):
    """Ca marche pas mais c'est joli"""
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a, dtype=float)
    
    #On copie les bords de l'image original
    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]
    
    for y in range(1, y_a-1):
        for x in range(1, x_a-1):
                for c in range(3):
                    val = 0
                    matrice = np.array([a[y-1, x-1][c], a[y-1, x][c], a[y-1, x+1][c],
                                a[y, x-1][c], a[y, x+1][c],
                                a[y+1, x-1][c], a[y+1, x+1][c], a[y+1, x+1][c]])
                    for i in range(len(matrice)):    
                        if matrice[i] != a[y, x][c]:
                            val += 1 / (matrice[i] - a[y, x][c])
                    new_a[y, x][c] = val
    return new_a

def egalisation_exacte(image):
    pixels = image.flatten()
    nb_pixels = pixels.size
    nb_niveaux = 256
    pixels_par_niveau = nb_pixels // nb_niveaux

    indices_tries = np.argsort(pixels, kind="stable")

    image_egalisee = np.zeros_like(pixels, dtype=np.uint8)

    for niveau in range(nb_niveaux):
        debut = niveau * pixels_par_niveau
        fin = (niveau + 1) * pixels_par_niveau
        image_egalisee[indices_tries[debut:fin]] = niveau

    image_egalisee[indices_tries[fin:]] = 255

    
    image_egalisee = image_egalisee.reshape(image.shape)

    return image_egalisee

def erosion(a, forme=True):
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]

    for y in range(1, y_a-1):
        for x in range(1, x_a-1):
                if forme :
                    somme = np.sum([a[y-1, x], 
                        a[y, x-1], a[y, x], a[y, x+1],
                                    a[y, x+1]])
                else:
                    somme = np.sum([a[y-1, x-1], a[y-1, x], a[y-1, x+1],
                            a[y, x-1], a[y, x], a[y, x+1],
                            a[y+1, x-1], a[y, x+1], a[y+1, x+1]])
                if somme > 0:
                    new_a[y, x] = 255
                else:
                    new_a[y, x] = 0
    return new_a

def dilatation(a, forme=True):
    x_a, y_a = a.shape[1], a.shape[0]
    new_a = np.zeros_like(a)

    new_a[0, :]  = a[0, :]
    new_a[-1, :] = a[-1, :]
    new_a[:, 0]  = a[:, 0]
    new_a[:, -1] = a[:, -1]

    for y in range(1, y_a-1):
        for x in range(1, x_a-1):
                if forme :
                    somme = np.sum([a[y-1, x], 
                        a[y, x-1], a[y, x], a[y, x+1],
                                    a[y, x+1]])
                    mult = 5
                else:
                    somme = np.sum([a[y-1, x-1], a[y-1, x], a[y-1, x+1],
                            a[y, x-1], a[y, x], a[y, x+1],
                            a[y+1, x-1], a[y, x+1], a[y+1, x+1]])
                    mult = 9
                
                if somme < mult*255:
                    new_a[y, x] = 0
                else:
                    new_a[y, x] = 255
    return new_a


def calcul_LUT(rgb, m):
    if rgb > 210 :
        return m
    return rgb

def LUT_personaliser(a):
    x_a = a.shape[0]
    y_a = a.shape[1]
    new_a = np.zeros_like(a)
    r = np.mean(a[..., 0])
    g = np.mean(a[..., 1])
    b = np.mean(a[..., 2])
    for i in range(x_a):
        for j in range(y_a):
            outR = calcul_LUT(a[i][j][0], r)
            outG = calcul_LUT(a[i][j][1], g)
            outB = calcul_LUT(a[i][j][2], b)
            new_a[i][j] = np.array((outR, outG, outB))
    return new_a
    

#%% Test ici
if __name__ == "__main__" :
    # Exercice 1
    
    im_lena = np.array(Image.open("../Images_TP/Lena.jpg"))

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

    im_noise = np.array(Image.open("../Images_TP/noise.tif"))
    im_clown = np.array(Image.open("../Images_TP/clown.tif"))

    affiche_gray(im_noise)
    affiche_gray(fft(im_noise, "noise"))
    affiche_gray(fft_filter(im_noise, "noise"))

    affiche_gray(im_clown)
    affiche_gray(fft(im_clown, "clown"))
    affiche_gray(fft_filter(im_clown, "clown"))

    #%% Exercice 3

    im_noisy_lena = np.array(Image.open("../Images_TP/noisy_Lena.png").convert("RGB"))
    im_mercury = np.array(Image.open("../Images_TP/mercury.tif"))
    
    
    affiche_gray(im_mercury)
    mercury_median = filtre_median(im_mercury)
    affiche_gray(mercury_median, title="Après median")
    affiche_gray(normalisation(mercury_median, max=120), title="Après median puis normalisation")
    
    
    affiche(im_noisy_lena)
    im_noisy_lena_median = filtre_median(im_noisy_lena)
    affiche(im_noisy_lena_median)
    im_noisy_lena_median = filtre_median(im_noisy_lena)
    affiche(im_noisy_lena_median)
    im_noisy_lena_median = accentuation(im_noisy_lena_median)
    affiche(im_noisy_lena_median)

    #%% Exercice 4
    
    im_tour_eiffel = Image.open("../Images_TP/tour-eiffel.jpg").convert("RGB")
    im_france = Image.open("../Images_TP/France.png").convert("RGB")

    fusion(im_tour_eiffel, im_france)

    #%% Exercice 5

    im_lena = np.array(Image.open("../Images_TP/Lena.jpg"))
    im_grey_lena = greyscale(im_lena, False)

    affiche_gray(im_grey_lena)
    histogramme(im_grey_lena)

    im_grey_lena = np.array(ImageOps.equalize(Image.fromarray(im_grey_lena)))
    affiche_gray(im_grey_lena, "Égalisation")
    histogramme(im_grey_lena, "Égalisation")

    im_lena_eq_exact = egalisation_exacte(im_grey_lena)
    affiche_gray(im_lena_eq_exact, "Égalisation exacte")
    histogramme(im_lena_eq_exact, "Histogramme Égalisation exacte")


    #%% Exercice 6 

    im_texte_a_restaurer = np.array(Image.open("../Images_TP/texte_a_restaurer.png").convert("RGB"))

    affiche(im_texte_a_restaurer)
    
    im_texte_a_restaurer = seuillage(im_texte_a_restaurer)
    affiche_gray(im_texte_a_restaurer, "seuillage")

    im_texte_a_restaurer = dilatation(im_texte_a_restaurer)
    im_texte_a_restaurer = erosion(im_texte_a_restaurer)
    affiche_gray(im_texte_a_restaurer, "Ouverture")

    #%% Exercice 7 

    im_photo_a_restaurer = np.array(Image.open("../Images_TP/photo_a_restaurer.png").convert("RGB"))

    affiche(im_photo_a_restaurer)

    im_photo_a_restaurer = LUT_personaliser(im_photo_a_restaurer)
    affiche(im_photo_a_restaurer, "LUT personaliser")

    im_photo_a_restaurer = lissage_gros(im_photo_a_restaurer, 6)
    affiche(im_photo_a_restaurer, "lissage_gros")

    # im_photo_a_restaurer = accentuation(im_photo_a_restaurer)
    # affiche(im_photo_a_restaurer, "accentuation")

    

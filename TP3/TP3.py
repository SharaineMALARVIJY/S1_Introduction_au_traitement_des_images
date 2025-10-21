# Sharaine MALARVIJY 21206543
#%% Fonctions 

from PIL import Image
from matplotlib import pyplot as plt
import cv2
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

def clip_int8(rgb):
    return np.uint8(np.clip(rgb, 0, 255))

# Fonction exercice 1

def refocus(a, a1, a2, ordinal=4):
    '''Cherche dans des fenetres de ordinal*ordinal
    le gradient donc l'image est la plus net'''
    x_a, y_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    grad = rgb2gray(a)
    grad1 = rgb2gray(a1)
    grad2 = rgb2gray(a2)

    grad = np.abs(filters.laplace(grad))
    grad1 = np.abs(filters.laplace(grad1))
    grad2 = np.abs(filters.laplace(grad2))

    # grad = np.abs(filters.sobel(grad))
    # grad1 = np.abs(filters.sobel(grad1))
    # grad2 = np.abs(filters.sobel(grad2))

    for x in range(0, x_a, ordinal):
        for y in range(0, y_a, ordinal):
            x_end = min(x + ordinal, x_a)
            y_end = min(y + ordinal, y_a)

            s = np.sum(grad[x:x_end, y:y_end])
            s1 = np.sum(grad1[x:x_end, y:y_end])
            s2 = np.sum(grad2[x:x_end, y:y_end])

            max_grad = max(s, s1, s2)
            if s == max_grad:
                new_a[x:x_end, y:y_end] = a[x:x_end, y:y_end]
            elif s1 == max_grad:
                new_a[x:x_end, y:y_end] = a1[x:x_end, y:y_end]
            else :
                new_a[x:x_end, y:y_end] = a2[x:x_end, y:y_end]
    return new_a


def refocus_horizontal(a, a1, a2):
    y_a, x_a = a.shape[0], a.shape[1]
    new_a = np.zeros_like(a)

    grad = np.abs(filters.laplace(a))
    grad1 = np.abs(filters.laplace(a1))
    grad2 = np.abs(filters.laplace(a2))

    for y in range(y_a):
        s = np.sum(grad[y])
        s1 = np.sum(grad1[y])
        s2 = np.sum(grad2[y])

        max_grad = max(s, s1, s2)
        if s == max_grad:
            new_a[y] = a[y]
        elif s1 == max_grad:
            new_a[y] = a1[y]
        else:
            new_a[y] = a2[y]

    return new_a

# Fonction exercice 2

def mosaic_images(img_list):
    """
    Stitch a list of overlapping images into a single panorama.
    Uses OpenCV's built-in Stitcher.
    """
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    _, stitched = stitcher.stitch(img_list)

    return stitched

# Fonction exercice 3


def remove_path(image, path):
    h, w, c = image.shape
    new_image = np.zeros((h, w-1, c), dtype=image.dtype)

    for y in range(h-1):
        x = path[y][1]
        new_image[y] = np.delete(image[y], x, axis=0)

    return new_image

def affiche_path(im_path, path):
    im_path_aff= np.copy(im_path)
    for i in path:
        im_path_aff[i[0], i[1], :3] = [255, 0, 0]
    affiche(im_path_aff)

def image_path(a):
    x_max, y_max = a.shape[1], a.shape[0]
    all_paths = []
    grad = filters.sobel(a)

    for x_start in range(0, x_max, 10): # Pas de 10 pour réduire les calculs
        path = []
        next_point = [[0, x_start]]  #premier point
        path_energy = grad[0, x_start]
        for i in range(y_max-1):
            y, x = next_point[i][0], next_point[i][1]
           
            if x == 0:
                voisin_bas = ([grad[y+1, x], 256, grad[y+1, x+1]])
            elif x == x_max-1:
                voisin_bas = ([grad[y+1, x], grad[y+1, x-1], 256])
            else:
                voisin_bas = ([grad[y+1, x], grad[y+1, x-1], grad[y+1, x+1]])

            voisin_min = np.argmin(voisin_bas)
            if voisin_min==0 :
                path.append([y+1, x])
                path_energy += grad[y+1, x]
            elif voisin_min==1:
                path.append([y+1, x-1])
                path_energy += grad[y+1, x-1]
            else : #voisin_min==2
                path.append([y+1, x+1])
                path_energy += grad[y+1, x+1]

            next_point.append(path[-1])

        all_paths.append((path, path_energy))
    
    min_path = min(all_paths, key=lambda p: p[1])[0]
    return min_path

def resizing(im, nb_colonne_retirer = 10, aff_path=False):
    im_path = np.copy(im)

    for _ in range(nb_colonne_retirer):
        path = image_path(rgb2gray(im_path))     
        if aff_path : affiche_path(im_path, path)
        im_path = remove_path(im_path, path)

    return im_path

# Fonction exercice 4

def blur_region(image, x_start, x_end, y_start, y_end, sigma=5):
    blurred_image = image.copy()
    region = image[y_start:y_end, x_start:x_end]
    blurred_region = filters.gaussian(region, sigma=sigma, channel_axis=-1)

    blurred_region = clip_int8(blurred_region * 255)
    
    blurred_image[y_start:y_end, x_start:x_end] = blurred_region
    return blurred_image 

# Fonction exercice 5
def focus_tour_eiffel(im):
    zones = [
    (0, 150, 10),   # (x_start, x_end, sigma)
    (150, 340, 5),
    (450, -150, 5),  
    (-150, -1, 10),
    ]

    new_im = im.copy()

    for x_start, x_end, sigma in zones:
        blurred_zone = blur_region(im, x_start=x_start, x_end=x_end, y_start=0, y_end=-1, sigma=sigma)
        new_im[:, x_start:x_end] = blurred_zone[:, x_start:x_end]
    return new_im

#%% Test ici
if __name__ == "__main__" :
    # Exercice 1
    
    im_refocus_1 = np.array(Image.open("../Images_TP/Refocus_1.png").convert("RGB"))
    im_refocus_2 = np.array(Image.open("../Images_TP/Refocus_2.png").convert("RGB"))
    im_refocus_3 = np.array(Image.open("../Images_TP/Refocus_3.png").convert("RGB"))
        
    im_refocus_1_gray = np.array(Image.open("../Images_TP/Refocus_1.png").convert("L"))
    im_refocus_2_gray = np.array(Image.open("../Images_TP/Refocus_2.png").convert("L"))
    im_refocus_3_gray = np.array(Image.open("../Images_TP/Refocus_3.png").convert("L"))

    affiche_3(im_refocus_1, im_refocus_2, im_refocus_3)

    grad = np.abs(filters.laplace((im_refocus_1_gray)))
    grad1 = np.abs(filters.laplace((im_refocus_2_gray)))
    grad2 = np.abs(filters.laplace((im_refocus_3_gray)))

    # grad = filters.sobel(im_refocus_1_gray)
    # grad1 = filters.sobel(im_refocus_2_gray)
    # grad2 = filters.sobel(im_refocus_3_gray)

    affiche_3(grad, grad1, grad2)

    fenetre = 4
    affiche(refocus(im_refocus_1, im_refocus_2, im_refocus_3, fenetre), title=f"Refocus automatique avec fenetre {fenetre}x{fenetre}")
    affiche(refocus_horizontal(im_refocus_1, im_refocus_2, im_refocus_3), title="Refocus automatique horizontal")

    # Fait de façon manuel
    claire3 = im_refocus_3[:125, :]
    claire2 = im_refocus_2[125:200, :]
    claire1 = im_refocus_1[200:,:]

    # affiche(claire3)
    # affiche(claire2)
    # affiche(claire1)

    new_a = np.zeros_like(im_refocus_1)
    new_a[200:,:] = claire1
    new_a[125:200, :] = claire2
    new_a[:125, :] = claire3
    affiche(new_a, title="Refocus manuel")



    #%% Exercice 2


    im_mosaic_1 = np.array(Image.open("../Images_TP/Mosaic_1.png").convert("RGB"))
    im_mosaic_2 = np.array(Image.open("../Images_TP/Mosaic_2.png").convert("RGB"))

    affiche(im_mosaic_1)
    affiche(im_mosaic_2)

    stitched = mosaic_images([im_mosaic_1, im_mosaic_2])

    affiche(stitched, title="Mosaïque panoramique (stitching)")



    #%% Exercice 3

    im_resize = np.array(Image.open("../Images_TP/Resize.png").convert("RGB"))
    im_resize_gray = np.array(Image.open("../Images_TP/Resize.png").convert("L"))

    affiche(im_resize, "Avant resizing")

    grad = filters.sobel(im_resize_gray)
    plt.figure()
    plt.title("Heat map")
    plt.imshow(grad, cmap="hot")
    

    im_resized = resizing(im_resize, nb_colonne_retirer = 130, aff_path=False)
    affiche(im_resized, "Après resizing")
    print("Axe x avant resizing :", len(im_resize[1]))
    print("Axe x après resizing :", len(im_resized[1]))



    #%% Exercice 4

    im_profondeur = np.array(Image.open("../Images_TP/Profondeur.png").convert("RGB"))

    grad = np.abs(filters.laplace((im_profondeur))) 

    new_im = blur_region(im_profondeur, x_start=0, x_end=300, y_start=0, y_end=-1, sigma=5)
    new_im = blur_region(im_profondeur, x_start=0, x_end=400, y_start=0, y_end=-1, sigma=5)
    new_im = blur_region(new_im, x_start=400, x_end=500, y_start=0, y_end=250, sigma=5)
    

    # Display results
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(im_profondeur)
    plt.subplot(1,2,2)
    plt.title("Flou de profondeur")
    plt.imshow(new_im)
    plt.show()

    #%% Exercice 5 

    im_tour_eiffel = np.array(Image.open("../Images_TP/tour-eiffel.jpg").convert("RGB"))

    affiche(im_tour_eiffel)

    affiche(focus_tour_eiffel(im_tour_eiffel))


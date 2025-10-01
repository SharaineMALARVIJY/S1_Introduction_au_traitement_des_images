from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

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
    return Image.fromarray(new_a)
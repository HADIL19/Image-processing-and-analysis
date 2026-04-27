import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
#Étape 1 : Charger et afficher une image en niveaux de gris 


#charger une image en niveau de gris
img_lena = cv2.imread('lena.webp', cv2.IMREAD_GRAYSCALE)
img_lung = cv2.imread('lung.webp', cv2.IMREAD_GRAYSCALE)
#cv2.IMREAD_GRAYSCALE convertit l’image en niveaux de gris 

#afficher l'image en niveau de gris
if img_lena is not None:
    print(img_lena.shape)
    cv2.imshow('Lena', img_lena)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not found.")
if img_lung is not None:
    print(img_lung.shape)
    cv2.imshow('Lung', img_lung)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not found.")
    
#🟢 Étape 2 : Calcul de l’histogramme 
def calcul_histogramme(image):
    hist = np.zeros(256)  # 256 niveaux de gris

    for i in range(image.shape[0]):      # lignes
        for j in range(image.shape[1]):  # colonnes
            pixel = image[i, j]
            hist[pixel] += 1

    return hist

# Calcul histogrammes
hist_lena = calcul_histogramme(img_lena)
hist_lung = calcul_histogramme(img_lung)
# Affichage histogrammes
plt.plot(hist_lena)
plt.title("Histogramme Lena")
plt.show()

plt.plot(hist_lung)
plt.title("Histogramme Lung")
plt.show()
#🟢 Étape 3 : Redimensionnement de l’image
scale_lung = cv2.resize(img_lung, (200, 200))
cv2.imwrite("scale_lung.jpg", scale_lung)

hist_scale = calcul_histogramme(scale_lung)

plt.plot(hist_scale)
plt.title("Histogramme Lung Redimensionnée")
plt.show()

#🟢 Étape 4 : Égalisation de l’histogramme
def histogramme_normalise(hist, image):
    total_pixels = image.shape[0] * image.shape[1]
    return hist / total_pixels

hist_norm = histogramme_normalise(hist_lena, img_lena)

plt.plot(hist_norm)
plt.title("Histogramme Normalisé")
plt.show()

#🟢 Étape 5 : Histogramme cumulé
def histogramme_cumule(hist):
    hist_cum = np.zeros(256)
    hist_cum[0] = hist[0]

    for i in range(1, 256):
        hist_cum[i] = hist_cum[i-1] + hist[i]

    return hist_cum

hist_cum = histogramme_cumule(hist_lena)

plt.plot(hist_cum)
plt.title("Histogramme Cumulé")
plt.show()
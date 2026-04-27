"""
USTHB - Faculté d'Informatique
Module : Traitement et Analyse d'Images | M1 BIOINFO
Série TP N°5 - Segmentation d'images (Partie 1)
Auteur : img_med_TP5.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. CHARGEMENT ET CONVERSION EN NIVEAUX DE GRIS
# ==============================================================
image_path = "Flower.webp"   # remplacer par "Flower.jpg" si nécessaire
image_color = cv2.imread(image_path)

if image_color is None:
    raise FileNotFoundError(f"Image introuvable : {image_path}")

# Conversion BGR -> RGB pour affichage correct avec matplotlib
image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Conversion en niveaux de gris
gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
print(f"Image chargée — taille : {gray.shape[1]}x{gray.shape[0]} pixels")

# ==============================================================
# 2. CALCUL DU GRADIENT AVEC LE FILTRE SOBEL
# ==============================================================
ksize = 3   # Taille du noyau Sobel (essayer 3, 5, 7 pour la question 2)

# Gradients horizontal et vertical
Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

# Magnitude du gradient  G = sqrt(Gx² + Gy²)
magnitude = np.sqrt(Gx**2 + Gy**2)
magnitude_norm = np.uint8(255 * magnitude / magnitude.max())   # normalisation 0-255

# Direction du gradient  θ = arctan(Gy / Gx)  [en degrés]
direction = np.arctan2(Gy, Gx) * (180 / np.pi)

print(f"Magnitude — min: {magnitude.min():.1f}  max: {magnitude.max():.1f}")
print(f"Direction  — min: {direction.min():.1f}°  max: {direction.max():.1f}°")

# ==============================================================
# 3. SEUILLAGE SIMPLE
# ==============================================================
seuil_bas  = 50    # Essayer : 30, 50, 80, 120, 180
seuil_haut = 150   # Essayer : 80, 120, 150, 200

# Seuillage simple avec un seul seuil (seuil_bas)
_, contours_simple = cv2.threshold(magnitude_norm, seuil_bas, 255, cv2.THRESH_BINARY)

# ==============================================================
# 4. SEUILLAGE PAR HYSTÉRÉSIS (algorithme de Canny)
#    Utilise les deux seuils : seuil_bas et seuil_haut
# ==============================================================
contours_hysteresis = cv2.Canny(gray, seuil_bas, seuil_haut, apertureSize=ksize)

# ==============================================================
# 5. AFFICHAGE DES RÉSULTATS
# ==============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f"Détection de contours — Sobel ksize={ksize} | "
             f"Seuil bas={seuil_bas}, Seuil haut={seuil_haut}", fontsize=13)

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Image originale (couleur)")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Niveaux de gris")
axes[0, 1].axis("off")

axes[0, 2].imshow(magnitude_norm, cmap="hot")
axes[0, 2].set_title("Magnitude du gradient")
axes[0, 2].axis("off")

axes[1, 0].imshow(np.abs(Gx), cmap="gray")
axes[1, 0].set_title("Gradient horizontal Gx")
axes[1, 0].axis("off")

axes[1, 1].imshow(contours_simple, cmap="gray")
axes[1, 1].set_title(f"Seuillage simple (seuil={seuil_bas})")
axes[1, 1].axis("off")

axes[1, 2].imshow(contours_hysteresis, cmap="gray")
axes[1, 2].set_title(f"Seuillage par hystérésis\n(bas={seuil_bas}, haut={seuil_haut})")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("resultats_TP5.png", dpi=150)
plt.show()
print("Figure sauvegardée : resultats_TP5.png")

# ==============================================================
# 6. COMPARAISON DE DIFFÉRENTS SEUILS (Question 1)
# ==============================================================
seuils = [(30, 60), (50, 100), (80, 160), (120, 220)]

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
fig2.suptitle("Influence des seuils sur la détection par hystérésis", fontsize=13)

for i, (sb, sh) in enumerate(seuils):
    simple_i = cv2.threshold(magnitude_norm, sb, 255, cv2.THRESH_BINARY)[1]
    canny_i  = cv2.Canny(gray, sb, sh, apertureSize=ksize)

    axes2[0, i].imshow(simple_i, cmap="gray")
    axes2[0, i].set_title(f"Simple  seuil={sb}")
    axes2[0, i].axis("off")

    axes2[1, i].imshow(canny_i, cmap="gray")
    axes2[1, i].set_title(f"Hystérésis  ({sb},{sh})")
    axes2[1, i].axis("off")

plt.tight_layout()
plt.savefig("comparaison_seuils_TP5.png", dpi=150)
plt.show()
print("Figure sauvegardée : comparaison_seuils_TP5.png")

# ==============================================================
# 7. COMPARAISON DES TAILLES DE NOYAU SOBEL (Question 2)
# ==============================================================
ksizes = [3, 5, 7]

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("Influence de la taille du noyau Sobel (ksize)", fontsize=13)

for i, k in enumerate(ksizes):
    gx_k = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy_k = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag_k = np.sqrt(gx_k**2 + gy_k**2)
    mag_k_norm = np.uint8(255 * mag_k / mag_k.max())

    axes3[i].imshow(mag_k_norm, cmap="hot")
    axes3[i].set_title(f"ksize = {k}")
    axes3[i].axis("off")

plt.tight_layout()
plt.savefig("comparaison_ksize_TP5.png", dpi=150)
plt.show()
print("Figure sauvegardée : comparaison_ksize_TP5.png")

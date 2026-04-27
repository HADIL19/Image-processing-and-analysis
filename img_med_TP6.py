"""
============================================================
  TP6 - Segmentation d'Images (Partie 2)
  Hystérésis + Orientation, Otsu, Composantes Connexes
  Module : Traitement et Analyse d'Images - M1 BIOINFO
  USTHB - Faculté d'Informatique
============================================================

NOTIONS CLÉS :
- Hystérésis orientée : tient compte de la direction du gradient
  pour explorer les bons voisins (suppression des non-maxima).
- Seuillage simple    : binarisation à seuil fixe.
- Méthode d'Otsu      : trouve automatiquement le meilleur seuil.
- Composantes connexes (CC) : étiquette les régions connexes d'une image binaire.
📘 Explication complète du TP6

🔷 Section 1 — Hystérésis avec Orientation des Gradients
Le pipeline se fait en 3 étapes :
Étape 1 — Gradients Sobel
Gx détecte les bordures verticales (variations horizontales d'intensité)
Gy détecte les bordures horizontales (variations verticales d'intensité)
Étape 2 — Magnitude et Direction

Magnitude : M = sqrt(Gx² + Gy²) → force du contour
Direction : angle = arctan2(Gy, Gx) → quantifié en 4 angles (0°, 45°, 90°, 135°)

Étape 3 — Seuillage Hystérésis
CasDécisionM > seuil_hautContour fort → 255M < seuil_basIgnoré → 0Entre les deuxComparer avec les 2 voisins orientés → si M est le maximum local → 255, sinon → 0
Les voisins dépendent de la direction du gradient :

0° → gauche/droite, 90° → haut/bas, 45° / 135° → diagonales


🔷 Section 2 — Seuillage Simple
pythonbinaire = np.where(image > seuil, 255, 0)
Seuil choisi manuellement (ex. 127) en regardant l'histogramme. Simple mais sensible au choix du seuil.

🔷 Section 3 — Méthode d'Otsu
Trouve automatiquement le meilleur seuil en maximisant la variance inter-classe :
σ²_b(T) = w1 × w2 × (μ1 − μ2)²

w1, w2 = proportions de pixels dans chaque groupe
μ1, μ2 = moyennes des intensités de chaque groupe
On teste tous les T de 0 à 255 et on retient celui qui donne le σ²_b maximum

Le script affiche aussi la courbe de variance pour voir où se trouve le maximum.

🔷 Section 4 — Composantes Connexes (BFS)
Algorithme BFS (Breadth-First Search) avec 8-connectivité :

Parcourir tous les pixels
Si un pixel blanc (255) n'est pas encore étiqueté → lancer un BFS
Propager l'étiquette à tous ses voisins blancs non étiquetés (et leurs voisins, etc.)
Incrémenter le compteur d'étiquettes pour la prochaine région

Le résultat est une image colorée où chaque objet détecté a une couleur unique, avec un tableau des tailles de chaque région.

Note : Pour Flower.jpg, décommentez la ligne correspondante dans le dictionnaire IMAGES en bas du scrip

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


# ==============================================================
# Imports des fonctions du TP5
# ==============================================================

# Noyaux Sobel
SOBEL_GX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

SOBEL_GY = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)


def convoluer_float(image, noyau):
    """Convolution avec résultat float (voir TP4/TP5)."""
    taille = noyau.shape[0]
    demi = taille // 2
    H, W = image.shape
    image_pad = np.pad(image.astype(np.float64), demi, mode='reflect')
    resultat = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            resultat[i, j] = np.sum(image_pad[i:i+taille, j:j+taille] * noyau)
    return resultat


def calculer_gradients(image):
    Gx = convoluer_float(image, SOBEL_GX)
    Gy = convoluer_float(image, SOBEL_GY)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.degrees(np.arctan2(Gy, Gx))
    return Gx, Gy, magnitude, direction


def noyau_gaussien_simple(taille=5, sigma=1.0):
    """Génère un noyau gaussien normalisé."""
    import math
    centre = taille // 2
    noyau = np.zeros((taille, taille), dtype=np.float64)
    for x in range(taille):
        for y in range(taille):
            dx, dy = x - centre, y - centre
            noyau[x, y] = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return noyau / noyau.sum()


def lisser_image(image, taille=5, sigma=1.0):
    return convoluer_float(image, noyau_gaussien_simple(taille, sigma)).astype(np.uint8)


# ==============================================================
# PARTIE 1 : Hystérésis avec Orientation (Non-Maximum Suppression)
# ==============================================================

def quantifier_direction(direction_deg):
    """
    Quantifie la direction du gradient en 4 angles : 0°, 45°, 90°, 135°.

    Principe : on arrondit l'angle au multiple de 45° le plus proche.
    Cela permet de savoir dans quelle direction explorer les voisins.

    Table de quantification :
    - Entre -22.5° et  22.5° (ou 157.5° à 180°) → 0°   (horizontal)
    - Entre  22.5° et  67.5°                     → 45°  (diagonale ↗)
    - Entre  67.5° et 112.5°                     → 90°  (vertical)
    - Entre 112.5° et 157.5°                     → 135° (diagonale ↘)
    """
    # Ramener les angles dans [0, 180)
    angle = direction_deg % 180

    direction_quant = np.zeros_like(direction_deg)
    direction_quant[(angle >= 0)   & (angle <  22.5)]  = 0    # horizontal
    direction_quant[(angle >= 22.5) & (angle < 67.5)]  = 45   # diagonale ↗
    direction_quant[(angle >= 67.5) & (angle < 112.5)] = 90   # vertical
    direction_quant[(angle >= 112.5) & (angle < 157.5)] = 135  # diagonale ↘
    direction_quant[(angle >= 157.5)]                   = 0    # horizontal

    return direction_quant


def suppression_non_maxima(magnitude, direction_quant):
    """
    Suppression des Non-Maxima (NMS) : ne garder que les maxima locaux.

    Pour chaque pixel, on regarde ses deux voisins dans la DIRECTION
    du gradient. Si le pixel n'est pas le maximum local, on l'élimine.

    Exemple pour angle 0° (horizontal) :
    → On compare le pixel avec ses voisins gauche et droite.
    → Si pixel < gauche OU pixel < droite → pas un contour.

    Cela permet d'affiner les contours (les rendre d'un pixel de large).
    """
    H, W = magnitude.shape
    mag_supprimee = np.zeros((H, W), dtype=np.float64)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            angle = direction_quant[i, j]
            mag = magnitude[i, j]

            # Sélectionner les deux voisins selon l'orientation
            if angle == 0:      # horizontal : voisins gauche/droite
                v1 = magnitude[i, j - 1]
                v2 = magnitude[i, j + 1]
            elif angle == 45:   # diagonale ↗ : voisins haut-droite/bas-gauche
                v1 = magnitude[i - 1, j + 1]
                v2 = magnitude[i + 1, j - 1]
            elif angle == 90:   # vertical : voisins haut/bas
                v1 = magnitude[i - 1, j]
                v2 = magnitude[i + 1, j]
            else:               # 135° diagonal ↘ : voisins haut-gauche/bas-droite
                v1 = magnitude[i - 1, j - 1]
                v2 = magnitude[i + 1, j + 1]

            # Garder seulement si c'est un maximum local
            if mag >= v1 and mag >= v2:
                mag_supprimee[i, j] = mag
            # sinon reste à 0

    return mag_supprimee


def hysteresis_avec_orientation(image, seuil_bas, seuil_haut):
    """
    Pipeline complet de détection de contours avec orientation :
    1. Lissage gaussien.
    2. Calcul des gradients Sobel.
    3. Quantification des directions.
    4. Suppression des non-maxima.
    5. Seuillage par hystérésis.

    C'est l'algorithme de Canny simplifié !
    """
    # Étape 1 : Lissage
    image_lissee = lisser_image(image, 5, 1.0)

    # Étape 2 : Gradients
    Gx, Gy, magnitude, direction = calculer_gradients(image_lissee)

    # Étape 3 : Quantification des directions
    direction_quant = quantifier_direction(direction)

    # Étape 4 : Suppression des non-maxima
    magnitude_mince = suppression_non_maxima(magnitude, direction_quant)

    # Étape 5 : Hystérésis
    FORT, FAIBLE, FOND = 255, 50, 0
    carte = np.zeros_like(magnitude_mince, dtype=np.uint8)
    carte[magnitude_mince >= seuil_haut] = FORT
    carte[(magnitude_mince >= seuil_bas) & (magnitude_mince < seuil_haut)] = FAIBLE

    # Propagation depuis les contours forts
    file = deque(map(tuple, np.argwhere(carte == FORT)))
    voisins_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    H, W = carte.shape

    while file:
        i, j = file.popleft()
        for di, dj in voisins_8:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and carte[ni, nj] == FAIBLE:
                carte[ni, nj] = FORT
                file.append((ni, nj))

    carte[carte == FAIBLE] = FOND
    return carte


# ==============================================================
# PARTIE 2 : Seuillage Simple
# ==============================================================

def seuillage_simple_binaire(image, seuil):
    """
    Binarise l'image : pixel ≥ seuil → 255 (avant-plan), sinon 0 (fond).

    On choisit le seuil manuellement en inspectant l'histogramme :
    - Les vallées entre deux pics indiquent de bons seuils.
    """
    image_bin = np.zeros_like(image, dtype=np.uint8)
    image_bin[image >= seuil] = 255
    return image_bin


# ==============================================================
# PARTIE 2 : Méthode d'Otsu
# ==============================================================

def calculer_histogramme(image):
    """Calcule l'histogramme (voir TP1)."""
    hist = np.zeros(256, dtype=np.int64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    return hist


def otsu_seuillage(image):
    """
    Méthode d'Otsu : trouve automatiquement le seuil optimal.

    Objectif : maximiser la variance INTER-CLASSE (= variance entre les
    deux groupes : fond et objet).

    Formule de la variance inter-classe pour un seuil T :
        σ²_b(T) = w1(T) × w2(T) × (μ1(T) - μ2(T))²

    Où :
        w1 = proportion de pixels dans le groupe 1 (intensité ≤ T)
        w2 = proportion de pixels dans le groupe 2 (intensité > T)
        μ1 = moyenne d'intensité du groupe 1
        μ2 = moyenne d'intensité du groupe 2

    On teste tous les T de 0 à 255 et on garde celui qui maximise σ²_b.

    INTUITION : un bon seuil sépare deux groupes bien distincts.
    Plus les deux groupes sont compacts ET bien séparés, plus σ²_b est grande.
    """
    hist = calculer_histogramme(image)
    total_pixels = image.shape[0] * image.shape[1]

    # Probabilités de chaque niveau
    prob = hist / total_pixels

    meilleur_seuil = 0
    meilleure_variance = 0.0

    for T in range(1, 255):
        # Groupe 1 : pixels ≤ T
        w1 = np.sum(prob[:T + 1])
        # Groupe 2 : pixels > T
        w2 = np.sum(prob[T + 1:])

        if w1 == 0 or w2 == 0:
            continue

        # Moyennes des deux groupes
        mu1 = np.sum(np.arange(T + 1) * prob[:T + 1]) / w1
        mu2 = np.sum(np.arange(T + 1, 256) * prob[T + 1:]) / w2

        # Variance inter-classe
        variance_interclasse = w1 * w2 * (mu1 - mu2) ** 2

        if variance_interclasse > meilleure_variance:
            meilleure_variance = variance_interclasse
            meilleur_seuil = T

    print(f"[Otsu] Seuil optimal trouvé : T = {meilleur_seuil}")
    print(f"       Variance inter-classe maximale = {meilleure_variance:.4f}")

    # Appliquer le seuil
    image_bin = seuillage_simple_binaire(image, meilleur_seuil)
    return image_bin, meilleur_seuil


# ==============================================================
# PARTIE 3 : Étiquetage par Composantes Connexes
# ==============================================================

def composantes_connexes(image_binaire, connectivite=8):
    """
    Étiquette les régions connexes d'une image binaire.

    Paramètres :
        image_binaire : image où 255 = avant-plan, 0 = fond.
        connectivite  : 4 (voisins croix) ou 8 (voisins diagonaux inclus).

    Résultat : matrice d'étiquettes où chaque région connexe
               a un numéro unique (1, 2, 3, ...).

    ALGORITHME BFS (Breadth-First Search) :
    ┌─────────────────────────────────────────────────────┐
    │ 1. Créer une matrice d'étiquettes (zéros = non vu). │
    │ 2. Pour chaque pixel non visité de l'avant-plan :   │
    │    a. Créer une nouvelle étiquette.                  │
    │    b. BFS : explorer tous les voisins connexes.      │
    │    c. Assigner la même étiquette à tous.             │
    │ 3. Résultat : chaque région connexe a un entier uniq.│
    └─────────────────────────────────────────────────────┘
    """
    H, W = image_binaire.shape
    etiquettes = np.zeros((H, W), dtype=np.int32)
    compteur = 0  # compteur d'étiquettes

    # Voisins selon la connectivité choisie
    if connectivite == 4:
        voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connectivité
        voisins = [(-1,-1),(-1,0),(-1,1),
                   ( 0,-1),       ( 0,1),
                   ( 1,-1),( 1,0),( 1,1)]

    # Parcourir chaque pixel
    for i in range(H):
        for j in range(W):
            # Si pixel d'avant-plan non encore étiqueté
            if image_binaire[i, j] > 0 and etiquettes[i, j] == 0:
                compteur += 1
                etiquettes[i, j] = compteur

                # BFS pour propager l'étiquette aux voisins connexes
                file = deque([(i, j)])

                while file:
                    ci, cj = file.popleft()
                    for di, dj in voisins:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < H and 0 <= nj < W
                                and image_binaire[ni, nj] > 0
                                and etiquettes[ni, nj] == 0):
                            etiquettes[ni, nj] = compteur
                            file.append((ni, nj))

    print(f"[CC] Nombre de composantes connexes trouvées : {compteur}")
    return etiquettes, compteur


def afficher_composantes(etiquettes, n_composantes):
    """
    Affiche les composantes connexes avec des couleurs aléatoires.
    Chaque région a une couleur unique.
    """
    H, W = etiquettes.shape
    image_couleur = np.zeros((H, W, 3), dtype=np.uint8)

    np.random.seed(42)  # pour la reproductibilité
    couleurs = np.random.randint(50, 255, (n_composantes + 1, 3), dtype=np.uint8)
    couleurs[0] = [0, 0, 0]  # fond = noir

    for etiquette in range(1, n_composantes + 1):
        masque = etiquettes == etiquette
        image_couleur[masque] = couleurs[etiquette]

    return image_couleur


# ==============================================================
# PROGRAMME PRINCIPAL
# ==============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TP6 - Segmentation : Otsu + CC")
    print("=" * 60)

    # Chargement des images
    flower  = cv2.imread('Flower.jpg',  cv2.IMREAD_GRAYSCALE)
    objects = cv2.imread('Objects.jpg', cv2.IMREAD_GRAYSCALE)

    # Si images non trouvées, créer des images synthétiques
    if flower is None:
        print("[INFO] Création image synthétique 'flower'")
        flower = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(flower, (100, 100), 70, 200, -1)
        for angle in range(0, 360, 45):
            x = int(100 + 90 * np.cos(np.radians(angle)))
            y = int(100 + 90 * np.sin(np.radians(angle)))
            cv2.circle(flower, (x, y), 20, 150, -1)
        cv2.GaussianBlur(flower, (5, 5), 0)

    if objects is None:
        print("[INFO] Création image synthétique 'objects'")
        objects = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(objects, (20,  20),  (80,  80),  200, -1)
        cv2.rectangle(objects, (100, 50),  (160, 120), 200, -1)
        cv2.circle   (objects, (220, 80),  35,         200, -1)
        cv2.circle   (objects, (120, 160), 25,         200, -1)
        cv2.rectangle(objects, (10,  140), (60, 190),  200, -1)

    print(f"[OK] Flower : {flower.shape}")
    print(f"[OK] Objects: {objects.shape}")

    # ─── PARTIE 1 : Hystérésis avec orientation ───────────────
    print("\n>>> PARTIE 1 : Hystérésis avec orientation (Canny simplifié)")

    contours_flower = hysteresis_avec_orientation(flower, seuil_bas=20, seuil_haut=60)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(flower,          cmap='gray'); axes[0].set_title("Flower originale"); axes[0].axis('off')
    axes[1].imshow(contours_flower, cmap='gray'); axes[1].set_title("Contours (NMS + Hystérésis)"); axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # ─── PARTIE 2 : Seuillage Simple ──────────────────────────
    print("\n>>> PARTIE 2a : Seuillage simple")

    for img, nom in [(flower, "Flower"), (objects, "Objects")]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title(f"{nom} original"); axes[0].axis('off')

        for idx, seuil in enumerate([64, 128, 192]):
            img_bin = seuillage_simple_binaire(img, seuil)
            axes[idx+1].imshow(img_bin, cmap='gray')
            axes[idx+1].set_title(f"Seuil = {seuil}")
            axes[idx+1].axis('off')

        plt.suptitle(f"Seuillage Simple - {nom}")
        plt.tight_layout()
        plt.show()

    # ─── PARTIE 2b : Méthode d'Otsu ───────────────────────────
    print("\n>>> PARTIE 2b : Méthode d'Otsu")

    images_otsu = {}
    for img, nom in [(flower, "Flower"), (objects, "Objects")]:
        print(f"\n[Otsu] Traitement de {nom}...")
        img_otsu, seuil_otsu = otsu_seuillage(img)
        images_otsu[nom] = img_otsu

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img,      cmap='gray'); axes[0].set_title(f"{nom} original"); axes[0].axis('off')
        axes[1].imshow(img_otsu, cmap='gray'); axes[1].set_title(f"Otsu (T={seuil_otsu})"); axes[1].axis('off')

        # Histogramme avec ligne de seuil
        hist = calculer_histogramme(img)
        axes[2].bar(range(256), hist, color='gray', width=1)
        axes[2].axvline(x=seuil_otsu, color='red', linewidth=2, label=f"Seuil Otsu = {seuil_otsu}")
        axes[2].set_title("Histogramme + seuil Otsu")
        axes[2].legend()

        plt.suptitle(f"Otsu - {nom}")
        plt.tight_layout()
        plt.show()

    # ─── PARTIE 3 : Composantes Connexes ──────────────────────
    print("\n>>> PARTIE 3 : Étiquetage par Composantes Connexes")

    for nom, img_bin in images_otsu.items():
        print(f"\n[CC] Traitement de {nom} binarisé par Otsu...")

        # 4-connectivité
        etiq_4, n4 = composantes_connexes(img_bin, connectivite=4)
        couleurs_4  = afficher_composantes(etiq_4, n4)

        # 8-connectivité
        etiq_8, n8 = composantes_connexes(img_bin, connectivite=8)
        couleurs_8  = afficher_composantes(etiq_8, n8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_bin,    cmap='gray');  axes[0].set_title(f"Binaire Otsu\n{nom}"); axes[0].axis('off')
        axes[1].imshow(couleurs_4);               axes[1].set_title(f"CC 4-connexité\n{n4} régions"); axes[1].axis('off')
        axes[2].imshow(couleurs_8);               axes[2].set_title(f"CC 8-connexité\n{n8} régions"); axes[2].axis('off')
        plt.suptitle(f"Composantes Connexes - {nom}")
        plt.tight_layout()
        plt.show()

    print("""
    RÉSUMÉ DES CONCEPTS - TP6 :
    ────────────────────────────

    Hystérésis orientée (Canny simplifié) :
      → La suppression des non-maxima affine les contours à 1 pixel.
      → L'orientation du gradient guide quels voisins comparer.

    Seuillage Simple vs Otsu :
      → Simple : seuil choisi manuellement (subjectif).
      → Otsu   : seuil calculé automatiquement en maximisant
                 la séparation entre fond et objet.
      → Otsu est robuste pour des images bimodales (2 pics distincts
        dans l'histogramme).

    Composantes Connexes :
      → 4-connexité  : voisins haut/bas/gauche/droite (plus strict).
      → 8-connexité  : + diagonales (regroupe plus de pixels).
      → Utilité : compter des objets, mesurer leur taille, les localiser.
      → Base de nombreuses applications médicales (comptage cellulaire).
    """)

    print("\n[TP6 TERMINÉ]")

"""
============================================================
  TP3 - Traitement d'images couleurs
  USTHB - M1 BIOINFO
  Auteur : Script complet - Quantification couleur + Histogramme
============================================================

Ce script couvre :
  1. Quantification couleur par K-means (implémenté manuellement)
  2. Quantification couleur par Median-Cut
  3. Calcul et visualisation de l'histogramme des couleurs
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==============================================================
# SECTION 0 : Chargement et conversion de l'image
# ==============================================================

def charger_image(chemin):
    """
    Charge l'image avec OpenCV (format BGR par défaut)
    et la convertit en RGB pour un affichage correct avec Matplotlib.

    Pourquoi convertir BGR -> RGB ?
    OpenCV stocke les canaux dans l'ordre Bleu-Vert-Rouge (BGR),
    alors que Matplotlib attend Rouge-Vert-Bleu (RGB).
    Sans conversion, les couleurs de l'image seraient inversées
    (ex : les tons chauds deviendraient froids et vice-versa).
    """
    img_bgr = cv2.imread(chemin, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# ==============================================================
# SECTION 1 : K-MEANS MANUEL
# ==============================================================

def initialiser_centres(pixels, k):
    """
    Étape a) Initialisation : choisir K centres aléatoirement parmi les pixels.
    On utilise np.random.choice pour sélectionner k indices uniques.
    """
    indices = np.random.choice(len(pixels), k, replace=False)
    return pixels[indices].astype(np.float64)


def assigner_clusters(pixels, centres):
    """
    Étape b) Assignation : pour chaque pixel, trouver le centre le plus proche.
    La distance utilisée est la distance euclidienne dans l'espace RGB (R, G, B).

    Pour chaque pixel p et chaque centre c :
      distance = sqrt((p[R]-c[R])² + (p[G]-c[G])² + (p[B]-c[B])²)

    On retourne un tableau d'indices de cluster (un entier par pixel).
    """
    # distances shape: (nb_pixels, k)
    distances = np.linalg.norm(pixels[:, np.newaxis] - centres[np.newaxis, :], axis=2)
    return np.argmin(distances, axis=1)


def mettre_a_jour_centres(pixels, labels, k):
    """
    Étape c) Mise à jour : recalculer chaque centre comme la moyenne
    des pixels qui lui sont assignés.
    Si un cluster est vide, on réinitialise son centre aléatoirement.
    """
    nouveaux_centres = np.zeros((k, 3), dtype=np.float64)
    for i in range(k):
        membres = pixels[labels == i]
        if len(membres) > 0:
            nouveaux_centres[i] = membres.mean(axis=0)
        else:
            # Cluster vide : on prend un pixel aléatoire pour éviter NaN
            nouveaux_centres[i] = pixels[np.random.randint(len(pixels))]
    return nouveaux_centres


def kmeans_manuel(image, k, max_iter=20, tolerance=1.0):
    """
    Algorithme K-means complet (sans sklearn ni cv2.kmeans).

    Paramètres :
      image    : image RGB (H x W x 3)
      k        : nombre de couleurs souhaité
      max_iter : nombre maximum d'itérations
      tolerance: seuil de convergence (déplacement maximal des centres)

    Retourne l'image quantifiée (H x W x 3).

    Étapes :
      a) Initialisation aléatoire des K centres
      b) Assignation de chaque pixel au centre le plus proche
      c) Mise à jour des centres (moyenne des pixels du cluster)
      d) Répétition jusqu'à convergence ou max_iter
    """
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)

    # a) Initialisation
    centres = initialiser_centres(pixels, k)
    print(f"  K-means démarré avec k={k}, max_iter={max_iter}")

    for iteration in range(max_iter):
        # b) Assignation
        labels = assigner_clusters(pixels, centres)

        # c) Mise à jour
        anciens_centres = centres.copy()
        centres = mettre_a_jour_centres(pixels, labels, k)

        # d) Vérification de la convergence
        deplacement = np.max(np.linalg.norm(centres - anciens_centres, axis=1))
        print(f"    Itération {iteration+1} | déplacement max des centres : {deplacement:.4f}")
        if deplacement < tolerance:
            print(f"  Convergence atteinte à l'itération {iteration+1}")
            break

    # Reconstruction de l'image : remplacer chaque pixel par la couleur de son centre
    pixels_quantifies = centres[labels].astype(np.uint8)
    image_quantifiee = pixels_quantifies.reshape(h, w, 3)
    return image_quantifiee, centres.astype(np.uint8)


# ==============================================================
# SECTION 2 : MEDIAN-CUT MANUEL
# ==============================================================

def median_cut(pixels, profondeur):
    """
    Algorithme Median-Cut (récursif).

    Principe :
      1. Trouver le canal (R, G ou B) avec la plus grande plage de valeurs.
      2. Trier les pixels selon ce canal.
      3. Diviser en deux moitiés égales (coupure à la médiane).
      4. Répéter récursivement jusqu'à atteindre la profondeur souhaitée.
      5. Chaque feuille de la récursion donne une couleur = moyenne du groupe.

    Paramètre profondeur : le nombre de couleurs final = 2^profondeur.
    """
    if profondeur == 0 or len(pixels) == 0:
        # Feuille : couleur représentative = moyenne du groupe
        return [pixels.mean(axis=0).astype(np.uint8)]

    # 1. Canal avec la plus grande plage
    plages = pixels.max(axis=0) - pixels.min(axis=0)
    canal = np.argmax(plages)  # 0=R, 1=G, 2=B

    # 2. Tri selon ce canal
    pixels_tries = pixels[pixels[:, canal].argsort()]

    # 3. Coupure à la médiane
    milieu = len(pixels_tries) // 2

    # 4. Récursion sur les deux moitiés
    palette = (median_cut(pixels_tries[:milieu], profondeur - 1) +
               median_cut(pixels_tries[milieu:], profondeur - 1))
    return palette


def quantifier_median_cut(image, nb_couleurs):
    """
    Applique Median-Cut à l'image et retourne l'image quantifiée.

    nb_couleurs doit être une puissance de 2 (4, 8, 16, 32...).
    On calcule automatiquement la profondeur = log2(nb_couleurs).
    """
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)

    # Calcul de la profondeur de récursion
    profondeur = int(np.log2(nb_couleurs))
    print(f"  Median-Cut : {nb_couleurs} couleurs (profondeur={profondeur})")

    # Construction de la palette
    palette = np.array(median_cut(pixels, profondeur))  # (nb_couleurs, 3)

    # Assignation : chaque pixel prend la couleur la plus proche dans la palette
    distances = np.linalg.norm(pixels[:, np.newaxis] - palette[np.newaxis, :], axis=2)
    labels = np.argmin(distances, axis=1)

    pixels_quantifies = palette[labels].astype(np.uint8)
    image_quantifiee = pixels_quantifies.reshape(h, w, 3)
    return image_quantifiee, palette.astype(np.uint8)


# ==============================================================
# SECTION 3 : HISTOGRAMME DES COULEURS
# ==============================================================

def calculer_histogramme(image_quantifiee):
    """
    Calcule l'histogramme des couleurs d'une image quantifiée.

    Étapes :
      1. Récupérer toutes les couleurs uniques de l'image.
      2. Compter le nombre de pixels pour chaque couleur.
      3. Normaliser : diviser par le nombre total de pixels → proportions entre 0 et 1.

    Retourne :
      couleurs_uniques : tableau (N, 3) des couleurs distinctes
      proportions      : tableau (N,)  de la proportion de chaque couleur
    """
    h, w, _ = image_quantifiee.shape
    pixels = image_quantifiee.reshape(-1, 3)
    nb_total = len(pixels)

    # Couleurs uniques et leur nombre d'occurrences
    couleurs_uniques, comptes = np.unique(pixels, axis=0, return_counts=True)

    # Normalisation
    proportions = comptes / nb_total

    # Tri par proportion décroissante pour un affichage lisible
    ordre = np.argsort(proportions)[::-1]
    couleurs_uniques = couleurs_uniques[ordre]
    proportions = proportions[ordre]

    return couleurs_uniques, proportions


def afficher_histogramme(couleurs_uniques, proportions, titre="Histogramme des couleurs"):
    """
    Affiche l'histogramme couleur :
      - Chaque barre correspond à une couleur unique de la palette.
      - La hauteur de la barre = proportion de pixels ayant cette couleur.
      - La couleur de la barre = la couleur réelle (en RGB normalisé 0-1).
    """
    fig, ax = plt.subplots(figsize=(max(10, len(couleurs_uniques) * 0.5), 5))

    x = np.arange(len(couleurs_uniques))
    largeur = 0.8

    for i, (couleur, proportion) in enumerate(zip(couleurs_uniques, proportions)):
        # Normaliser RGB de [0,255] vers [0,1] pour Matplotlib
        couleur_norm = couleur / 255.0
        ax.bar(i, proportion, width=largeur, color=couleur_norm, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in couleurs_uniques],
        rotation=90, fontsize=7
    )
    ax.set_ylabel("Proportion de pixels")
    ax.set_xlabel("Couleurs (code hexadécimal)")
    ax.set_title(titre)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


# ==============================================================
# SECTION 4 : PROGRAMME PRINCIPAL
# ==============================================================

if __name__ == "__main__":

    # ── Chargement ───────────────────────────────────────────
    CHEMIN_IMAGE = "lena.webp"   # Adaptez si nécessaire (lena.jpg, lena.png, etc.)
    print("=== Chargement de l'image ===")
    image = charger_image(CHEMIN_IMAGE)
    print(f"  Dimensions : {image.shape[1]}x{image.shape[0]} pixels")

    # ── K-means avec différentes valeurs de K ────────────────
    valeurs_k = [2, 4, 8, 16]

    print("\n=== K-MEANS ===")
    fig_kmeans, axes = plt.subplots(1, len(valeurs_k) + 1, figsize=(4 * (len(valeurs_k) + 1), 4))
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    images_kmeans = {}
    for i, k in enumerate(valeurs_k):
        print(f"\n  → K = {k}")
        img_q, centres = kmeans_manuel(image, k=k, max_iter=15)
        images_kmeans[k] = img_q
        axes[i + 1].imshow(img_q)
        axes[i + 1].set_title(f"K-means k={k}")
        axes[i + 1].axis("off")

    fig_kmeans.suptitle("Quantification K-means — Comparaison du nombre de couleurs", fontsize=13)
    plt.tight_layout()
    plt.savefig("kmeans_comparaison.png", dpi=120)
    plt.show()

    # ── Median-Cut avec différents nombres de couleurs ───────
    valeurs_mc = [4, 8, 16, 32]  # doivent être des puissances de 2

    print("\n=== MEDIAN-CUT ===")
    fig_mc, axes2 = plt.subplots(1, len(valeurs_mc) + 1, figsize=(4 * (len(valeurs_mc) + 1), 4))
    axes2[0].imshow(image)
    axes2[0].set_title("Image originale")
    axes2[0].axis("off")

    images_mc = {}
    for i, nb in enumerate(valeurs_mc):
        print(f"\n  → {nb} couleurs")
        img_q, palette = quantifier_median_cut(image, nb_couleurs=nb)
        images_mc[nb] = img_q
        axes2[i + 1].imshow(img_q)
        axes2[i + 1].set_title(f"Median-Cut n={nb}")
        axes2[i + 1].axis("off")

    fig_mc.suptitle("Quantification Median-Cut — Comparaison du nombre de couleurs", fontsize=13)
    plt.tight_layout()
    plt.savefig("mediancut_comparaison.png", dpi=120)
    plt.show()

    # ── Histogrammes des images quantifiées ──────────────────
    print("\n=== HISTOGRAMMES ===")

    # Exemple avec K-means k=8
    k_choisi = 8
    print(f"\n  Histogramme K-means k={k_choisi}")
    img_km8 = images_kmeans[k_choisi]
    couleurs, proportions = calculer_histogramme(img_km8)
    print(f"  Nombre de couleurs uniques trouvées : {len(couleurs)}")
    fig_hist1 = afficher_histogramme(couleurs, proportions,
                                     titre=f"Histogramme K-means (k={k_choisi})")
    fig_hist1.savefig(f"histogramme_kmeans_k{k_choisi}.png", dpi=120)
    plt.show()

    # Exemple avec Median-Cut 16 couleurs
    nb_choisi = 16
    print(f"\n  Histogramme Median-Cut n={nb_choisi}")
    img_mc16 = images_mc[nb_choisi]
    couleurs2, proportions2 = calculer_histogramme(img_mc16)
    print(f"  Nombre de couleurs uniques trouvées : {len(couleurs2)}")
    fig_hist2 = afficher_histogramme(couleurs2, proportions2,
                                     titre=f"Histogramme Median-Cut (n={nb_choisi})")
    fig_hist2.savefig(f"histogramme_mediancut_n{nb_choisi}.png", dpi=120)
    plt.show()

    print("\n✅ Script terminé. Toutes les figures ont été sauvegardées.")
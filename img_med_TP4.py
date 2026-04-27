import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Chemins relatifs (Windows / Linux / Mac)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Chargement de l'image en niveaux de gris
# ─────────────────────────────────────────────
IMAGE_PATH = os.path.join(BASE_DIR, 'noisy_lena.webp')
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"\n[ERREUR] Image introuvable : {IMAGE_PATH}"
        "\n-> Verifiez que 'noisy_lena.webp' est dans le meme dossier que ce script."
    )

print(f"Image chargee : {image.shape[1]}x{image.shape[0]} px")


# ══════════════════════════════════════════════
# FONCTIONS
# ══════════════════════════════════════════════

def filtre_Gaussien(sigma: float, taille: int) -> np.ndarray:
    """
    Genere un noyau gaussien 2D normalise.
    G(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
    """
    center = taille // 2
    kernel = np.zeros((taille, taille), dtype=np.float64)
    for x in range(taille):
        for y in range(taille):
            dx = x - center
            dy = y - center
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * \
                           np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def sauvegarder(nom: str):
    chemin = os.path.join(OUTPUT_DIR, nom)
    plt.savefig(chemin, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> sauvegarde : {chemin}")


# ══════════════════════════════════════════════
# 1. FILTRE MOYENNEUR  (cv2.filter2D)
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("1. FILTRE MOYENNEUR")
print("=" * 50)

kernel_3 = np.ones((3, 3), dtype=np.float32) / 9
kernel_5 = np.ones((5, 5), dtype=np.float32) / 25
kernel_7 = np.ones((7, 7), dtype=np.float32) / 49

print("Noyau moyenneur 3x3 :\n", np.round(kernel_3, 4))

moy_3 = cv2.filter2D(image, -1, kernel_3)
moy_5 = cv2.filter2D(image, -1, kernel_5)
moy_7 = cv2.filter2D(image, -1, kernel_7)

fig1, axes = plt.subplots(1, 4, figsize=(18, 5))
fig1.suptitle("Filtre Moyenneur", fontsize=14, fontweight='bold')
for ax, img, title in zip(axes,
                           [image, moy_3, moy_5, moy_7],
                           ["Image bruitee", "Moyenneur 3x3",
                            "Moyenneur 5x5", "Moyenneur 7x7"]):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
sauvegarder('filtre_moyenneur.png')


# ══════════════════════════════════════════════
# 2. FILTRE GAUSSIEN  (cv2.filter2D + noyau manuel)
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. FILTRE GAUSSIEN")
print("=" * 50)

k_ref = filtre_Gaussien(sigma=1, taille=3)
print("Noyau Gaussien (sigma=1, 3x3) :\n", np.round(k_ref, 4))

configs = [(1.0, 3), (0.5, 3), (1.0, 5), (2.0, 5), (1.0, 7), (2.0, 7)]

fig2, axes = plt.subplots(3, 3, figsize=(15, 14))
fig2.suptitle("Filtre Gaussien - tailles et sigma varies", fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("Image bruitee")
axes[0, 0].axis('off')

gauss_results = {}
for plot_idx, (sigma, taille) in enumerate(configs, start=1):
    k   = filtre_Gaussien(sigma, taille).astype(np.float32)
    res = cv2.filter2D(image, -1, k)
    gauss_results[(sigma, taille)] = res
    if plot_idx <= 8:
        row, col = divmod(plot_idx, 3)
        axes[row, col].imshow(res, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f"sigma={sigma}, {taille}x{taille}")
        axes[row, col].axis('off')

plt.tight_layout()
sauvegarder('filtre_gaussien.png')


# ══════════════════════════════════════════════
# 3. FILTRE MEDIAN  (cv2.medianBlur)
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. FILTRE MEDIAN")
print("=" * 50)

med_3 = cv2.medianBlur(image, 3)
med_5 = cv2.medianBlur(image, 5)
med_7 = cv2.medianBlur(image, 7)

fig3, axes = plt.subplots(1, 4, figsize=(18, 5))
fig3.suptitle("Filtre Median", fontsize=14, fontweight='bold')
for ax, img, title in zip(axes,
                           [image, med_3, med_5, med_7],
                           ["Image bruitee", "Median 3x3",
                            "Median 5x5", "Median 7x7"]):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
sauvegarder('filtre_median.png')


# ══════════════════════════════════════════════
# 4. COMPARAISON GLOBALE (noyau 5x5)
# ══════════════════════════════════════════════
gauss_5 = gauss_results[(1.0, 5)]

fig4, axes = plt.subplots(1, 4, figsize=(18, 5))
fig4.suptitle("Comparaison des filtres (noyau 5x5)", fontsize=14, fontweight='bold')
for ax, img, title in zip(axes,
                           [image, moy_5, gauss_5, med_5],
                           ["Image bruitee", "Moyenneur 5x5",
                            "Gaussien 5x5 sigma=1", "Median 5x5"]):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
sauvegarder('comparaison_filtres.png')

print(f"\nTous les resultats sont dans : {OUTPUT_DIR}")

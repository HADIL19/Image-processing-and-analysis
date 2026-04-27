import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("low_cont_xray.jpg", cv2.IMREAD_GRAYSCALE)

    
#1. Translation de l’histogramme You add a constant to all pixels: I′=I+k
#❓ Answers
#k > 0 → image becomes brighter
#k < 0 → image becomes darker
#Limits:
#values < 0 → clipped to 0
#values > 255 → clipped to 255
def translation(image, k):
    img_trans = image.astype(np.int16) + k  # avoid overflow
    img_trans = np.clip(img_trans, 0, 255)  # keep valid range
    return img_trans.astype(np.uint8)

img_plus = translation(img, 50)   # brighter
img_minus = translation(img, -50) # darker 

plt.figure()
plt.title("Original Histogram")
plt.hist(img.flatten(), bins=256)

plt.figure()
plt.title("Translated +50")
plt.hist(img_plus.flatten(), bins=256)

plt.figure()
plt.title("Translated -50")
plt.hist(img_minus.flatten(), bins=256)

plt.show()

#2. Inversion de l’histogramme 💡 Formula I′=255−I 
#❓ Answers
#Bright areas become dark and vice versa
#It’s like a negative image

#✔ Useful in: -medical imaging (X-rays) - highlighting structures
def inversion(image):
    return 255 - image

img_inv = inversion(img)

plt.figure()
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.figure()
plt.title("Inverted")
plt.imshow(img_inv, cmap='gray')

plt.show()

#3. Expansion de Dynamique  💡 Formula I′=255(I−Imin)/(Imax−Imin)

def expansion(image):
    Imin = np.min(image)
    Imax = np.max(image)

    img_exp = (image - Imin) * 255 / (Imax - Imin)
    return img_exp.astype(np.uint8)

img_exp = expansion(img)

#Égalisation d’histogramme 💡 Formula I′=255∑j=0i(nj/N) where nj is the number of pixels with intensity j and N is the total number of pixels

def equalisation(image):
    hist, _ = np.histogram(image.flatten(), 256, [0,256])
    
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]

    MN = image.size

    cdf_norm = (cdf - cdf_min) / (MN - cdf_min) * 255
    cdf_norm = cdf_norm.astype(np.uint8)

    img_eq = cdf_norm[image]
    return img_eq

img_eq = equalisation(img)

plt.figure()
plt.title("Equalized Image")
plt.imshow(img_eq, cmap='gray')
plt.show()
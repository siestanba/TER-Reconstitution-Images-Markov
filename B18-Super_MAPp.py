import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

def ajouter_bruit(champ, sigma, nb_etats):
    bruit = np.random.normal(0, sigma, size=champ.shape)
    champ_bruite = np.round(champ + bruit).astype(int)
    champ_bruite = np.clip(champ_bruite, 0, nb_etats - 1)
    return champ_bruite

def generer_1000_bruitees(img, sigma, nb_etats, n_samples=1000):
    H, L = img.shape
    all_samples = np.zeros((n_samples, H, L), dtype=int)
    for k in tqdm(range(n_samples), desc="Génération des 1000 bruitages"):
        all_samples[k] = ajouter_bruit(img, sigma, nb_etats)
    return all_samples

def argmax_pixel_par_pixel(samples, nb_etats):
    H, L = samples[0].shape
    result = np.zeros((H, L), dtype=int)
    for i in range(H):
        for j in range(L):
            counts = np.bincount(samples[:, i, j], minlength=nb_etats)
            result[i, j] = np.argmax(counts)
    return result

def afficher(img, nb_etats):
    plt.imshow((img * (255 / (nb_etats - 1))).astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

# === Paramètres ===
chemin_image = "images/test1.png"
nb_etats = 5
sigma_bruit = 1.5
n_samples = 1000

# === Pipeline ===
img = charger_image_grayscale(chemin_image, nb_etats)
samples = generer_1000_bruitees(img, sigma_bruit, nb_etats, n_samples)
img_aggreg = argmax_pixel_par_pixel(samples, nb_etats)

# === Affichage ===
afficher(img_aggreg, nb_etats)
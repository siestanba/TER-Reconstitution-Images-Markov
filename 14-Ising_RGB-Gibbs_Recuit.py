import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def charger_image_rgb(path):
    img = Image.open(path).convert("RGB").resize((100, 100))
    return np.array(img)

def rgb_to_channels(img_rgb):
    return img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]  # R, G, B

def binariser_channel(channel, seuil=128):
    return (channel > seuil).astype(int)

def ajouter_bruit_channels(channels, p):
    bruitees = []
    for ch in channels:
        bruit = np.random.choice([0, 1], size=ch.shape)
        masque = np.random.rand(*ch.shape) < p
        bruitees.append(np.where(masque, bruit * 255, ch))
    return tuple(bruitees)

def cdf_locale_4(i, j, H, L, champ, etat):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = 0
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += -1 if etat == champ[ni, nj] else 1
    return energie

def gibbs_ising_recuit(champ, nb_iter):
    H, L = champ.shape
    champ = binariser_channel(champ)
    for k in tqdm(range(nb_iter), desc="Gibbs recuit simulé Ising"):
        T = 1 / np.log(2 + k)
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        energies = np.array([cdf_locale_4(i, j, H, L, champ, s) for s in [0, 1]])
        proba = np.exp(-energies / T)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice([0, 1], p=proba)
    return champ

def reconstituer_image(R, G, B):
    return np.stack([R, G, B], axis=2).astype(np.uint8)

def normaliser_channel(channel):
    return (channel * 255).astype(np.uint8)

def afficher_images(images, titres):
    n = len(images)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(20, 10))
    axs = axs.flatten()
    for ax, img, titre in zip(axs, images, titres):
        ax.imshow(img)
        ax.set_title(titre)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# === Paramètres ===
chemin_image = "images/RGB.png"
p_bruit = 0.3
nb_iter = 100000

# === Pipeline ===
img_rgb = charger_image_rgb(chemin_image)

# Bruit sur l'image RGB directement
R_raw, G_raw, B_raw = rgb_to_channels(img_rgb)
R_bruit, G_bruit, B_bruit = ajouter_bruit_channels((R_raw, G_raw, B_raw), p_bruit)

# Binarisation
R_bin = binariser_channel(R_bruit)
G_bin = binariser_channel(G_bruit)
B_bin = binariser_channel(B_bruit)

# Recuit simulé Ising
R_restaure = gibbs_ising_recuit(R_bruit.copy(), nb_iter)
G_restaure = gibbs_ising_recuit(G_bruit.copy(), nb_iter)
B_restaure = gibbs_ising_recuit(B_bruit.copy(), nb_iter)

img_final = reconstituer_image(
    normaliser_channel(R_restaure),
    normaliser_channel(G_restaure),
    normaliser_channel(B_restaure)
)

# === Affichage ===
afficher_images([
    img_rgb,
    reconstituer_image(R_bruit, G_bruit, B_bruit),
    normaliser_channel(R_bin), normaliser_channel(G_bin), normaliser_channel(B_bin),
    normaliser_channel(R_restaure), normaliser_channel(G_restaure), normaliser_channel(B_restaure),
    img_final
], [
    "Image originale RGB",
    "Image bruitée RGB",
    "Canal Rouge binarisé", "Canal Vert binarisé", "Canal Bleu binarisé",
    "Rouge restauré", "Vert restauré", "Bleu restauré",
    "Image finale restaurée (RGB)"
])

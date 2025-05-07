import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement et quantification de l'image ===
def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

# === Ajout de bruit : simulateur du modèle d'observation P(Y|X) ===
def ajouter_bruit(champ, p, nb_etats):
    bruit = np.random.choice(range(nb_etats), size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)

# === Calcul de l'énergie locale pour MAP (avec attache aux données) ===
def energie_locale_map(i, j, H, L, champ, etat, poids_aretes, observation, sigma2):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = ((etat - observation[i, j]) ** 2) / (2 * sigma2)
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

# === Estimateur MAP via descente locale (type ICM) ===
def estimateur_map(champ, observation, nb_iter, modele, sigma2):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="MAP"):
        i, j = np.random.randint(H), np.random.randint(L)
        local_energies = np.array([
            energie_locale_map(i, j, H, L, champ, s, modele['poids_aretes'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        champ[i, j] = np.argmin(local_energies)
    return champ

# === Estimateur MAP avec recuit simulé ===
def estimateur_map_recuit(champ, observation, nb_iter, modele, sigma2, t=1.0):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="MAP recuit simulé"):
        T = t / np.log(2 + k)
        i, j = np.random.randint(H), np.random.randint(L)
        local_energies = np.array([
            energie_locale_map(i, j, H, L, champ, s, modele['poids_aretes'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-local_energies / T)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
    return champ


# === Visualisation ===
def afficher_resultats_multi_map(img_init, img_bruitee, map1, map2, map3, nb_etats, taux, taux_base, taux_map1, taux_map2, taux_map3):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    titres = [f"Image originale \n(ε={taux:.2f})", f"Image bruitée \n(ε={taux_base:.2f})", f"MAP (β={beta1}, σ²={sigma1}) \n(ε={taux_map1:.2f})", f"MAP (β={beta2}, σ²={sigma2}) \n(ε={taux_map2:.2f})", f"MAP (β={beta3}, σ²={sigma3}) \n(ε={taux_map3:.2f})"]
    images = [img_init, img_bruitee, map1, map2, map3]
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for ax, titre, img in zip(axs, titres, images):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def taux_restauration(img_originale, img_restauree):
    pixels_corrects = np.sum(img_originale == img_restauree)
    total_pixels = img_originale.size
    taux = (pixels_corrects / total_pixels) * 100
    return taux


# === Paramètres ===
chemin_image = "images/test1.png"
nb_etats = 3
p_bruit = 0.3
nb_iter = 100000

img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
champ_init = img_bruitee.copy()

# MAP 1 : beta = 0.3, sigma2 = 1
beta1 = 1
sigma1 = 1
poids_aretes1 = {(a, b): beta1 * (-1 if a == b else 1) for a in range(nb_etats) for b in range(nb_etats)}
modele1 = {'nb_etats': nb_etats, 'poids_aretes': poids_aretes1, 'poids_sommets': [0] * nb_etats}
map1 = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele1, sigma1)

# MAP 2 : beta = 0.7, sigma2 = 1
beta2 = 0.2
sigma2 = 1
poids_aretes2 = {(a, b): beta2 * (-1 if a == b else 1) for a in range(nb_etats) for b in range(nb_etats)}
modele2 = {'nb_etats': nb_etats, 'poids_aretes': poids_aretes2, 'poids_sommets': [0] * nb_etats}
map2 = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele2, sigma2)

# MAP 3 : beta = 1.0, sigma2 = 2
beta3 = -1
sigma3 = 1
poids_aretes3 = {(a, b): beta3 * (-1 if a == b else 1) for a in range(nb_etats) for b in range(nb_etats)}
modele3 = {'nb_etats': nb_etats, 'poids_aretes': poids_aretes3, 'poids_sommets': [0] * nb_etats}
map3 = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele3, sigma3)

# Calcul des taux de restauration
taux = taux_restauration(img, img)
taux_base = taux_restauration(img, img_bruitee)
taux_map1 = taux_restauration(img, map1)
taux_map2 = taux_restauration(img, map2)
taux_map3 = taux_restauration(img, map3)


# Affichage final
afficher_resultats_multi_map(img, img_bruitee, map1, map2, map3, nb_etats, taux, taux_base, taux_map1, taux_map2, taux_map3)

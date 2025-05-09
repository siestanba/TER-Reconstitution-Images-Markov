import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    img_bin = np.floor(img_np / (256 / nb_etats)).astype(int)  # 0 ou 1
    return 2 * img_bin - 1  # transforme en -1 ou +1

def ajouter_bruit_gaussien_discret(champ, sigma):
    bruit = np.random.normal(0, sigma, size=champ.shape)
    champ_bruite = np.sign(champ + bruit)  # Garde -1 ou +1
    champ_bruite[champ_bruite == 0] = 1    # Remplace les 0 par +1 (par convention)
    return champ_bruite

def ajouter_bruit(champ, p):
    bruit = np.random.choice([-1, 1], size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)


def energie_locale_ising(i, j, H, L, champ, etat, beta, B):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    somme_voisins = 0
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            somme_voisins += champ[ni, nj]
    energie = -beta * etat * somme_voisins - B * etat
    return energie

def gibbs_ising(champ, nb_iter, beta, B):
    H, L = champ.shape
    for _ in tqdm(range(nb_iter), desc="Gibbs Ising"):
        i, j = np.random.randint(0, H), np.random.randint(0, L)
        energies = []
        for s in [-1, 1]:
            e = energie_locale_ising(i, j, H, L, champ, s, beta, B)
            energies.append(np.exp(-e))
        proba = energies / np.sum(energies)
        champ[i, j] = np.random.choice([-1, 1], p=proba)
    return champ

def metropolis_ising(champ, nb_iter, beta, B):
    H, L = champ.shape
    for _ in tqdm(range(nb_iter), desc="Metropolis Ising"):
        i, j = np.random.randint(0, H), np.random.randint(0, L)
        etat_actuel = champ[i, j]
        etat_nouveau = -etat_actuel
        E_actuel = energie_locale_ising(i, j, H, L, champ, etat_actuel, beta, B)
        E_nouveau = energie_locale_ising(i, j, H, L, champ, etat_nouveau, beta, B)
        delta_E = E_nouveau - E_actuel
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E):
            champ[i, j] = etat_nouveau
    return champ

def afficher_resultats(img_init, img_bruitee, gibbs, metro, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    
    # Calcul du taux de restauration pour les deux estimateurs
    taux = taux_restauration(img_init, img_init)
    taux_base = taux_restauration(img_init, img_bruitee)
    taux_gibbs = taux_restauration(img_init, gibbs)
    taux_metro = taux_restauration(img_init, metro)

    imgs = [img_init, gibbs, img_bruitee, metro]
    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Gibbs classique \n(ε={taux_gibbs:.2f})",
        f"Image bruitée \n(ε={taux_base:.2f})", f"Metropolis classique \n(ε={taux_metro:.2f})"
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i, (ax, titre, img) in enumerate(zip(axs, titres, imgs)):
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
chemin_image = "images/souris.png"
#chemin_image = "images/chat.jpg"
sigma_bruit = 1
p_bruit = 0.3
nb_etats = 2
nb_iter = 100000
beta = 2
B = 0.0  # champ externe nul


# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
#img_bruitee = ajouter_bruit_gaussien_discret(img, sigma_bruit)
img_bruitee = ajouter_bruit(img, p_bruit)
champ_gibbs = img_bruitee.copy()
champ_metro = img_bruitee.copy()

champ_gibbs = gibbs_ising(champ_gibbs, nb_iter, beta, B)
champ_metro = metropolis_ising(champ_metro, nb_iter, beta, B)

# Affichage des résultats
afficher_resultats(img, img_bruitee, champ_gibbs, champ_metro, nb_etats)